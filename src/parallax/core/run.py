from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parallax.core.agent import Agent

from parallax.core.errors import (
    LLMExecutionError,
    MaxIterationsError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from parallax.core.context import UsageBaseline
from parallax.core.context.shrink import (
    extract_token_limit,
    is_request_too_large,
    shrink_to_fit,
)
from parallax.core.llm import (
    GENERATION_ERROR_HINT,
    backoff_delay,
    extract_retry_delay,
    is_generation_error,
    is_retryable,
)
from parallax.core.types import AgentEvent, AgentResult, EventType, LLMResponse, Message

MAX_GENERATION_RETRIES = 4


class AgentRun:
    def __init__(
        self,
        agent: Agent,
        user_input: str,
        history: list[Message] | None = None,
    ):
        self.agent = agent
        self.user_input = user_input
        prior_messages = history or []
        filtered_prior = [m for m in prior_messages if m.role != "system"]
        self.messages = [
            Message(
                role="system",
                content=agent.context.build_system_prompt(agent.system_prompt),
            ),
            *filtered_prior,
            Message(role="user", content=user_input),
        ]
        self.iterations = 0
        self.run_id = uuid.uuid4().hex
        self._baseline: UsageBaseline | None = None
        self._event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def _emit(self, event_type: EventType, **data):
        """Push an event to the queue"""
        event = AgentEvent(type=event_type, data=data, run_id=self.run_id)
        await self._event_queue.put(event)

    async def _dispatch_tool(self, tool_call) -> Message:
        name = tool_call.name
        arguments = tool_call.arguments

        await self._emit(
            EventType.TOOL_STARTED,
            tool_call_id=tool_call.id,
            tool_name=name,
            arguments=arguments,
        )

        try:
            if tool_call.error:
                raise ToolError(tool_call.error)
            if name not in self.agent.tool_registry:
                raise ToolNotFoundError(f"Tool '{name}' not registered")
            func = self.agent.tool_registry[name]
            # Models often pass null for optional params; treat as "omitted".
            call_args = {k: v for k, v in arguments.items() if v is not None}
            result = await asyncio.wait_for(
                asyncio.to_thread(func, **call_args), timeout=30.0
            )
            await self._emit(
                EventType.TOOL_COMPLETED,
                tool_call_id=tool_call.id,
                tool_name=name,
                result=str(result),
            )

        except (ToolError, ToolNotFoundError) as e:
            result = f"Error: {e}"
            await self._emit(
                EventType.TOOL_FAILED,
                tool_call_id=tool_call.id,
                tool_name=name,
                error=str(result),
            )
        except asyncio.TimeoutError:
            result = f"Error: Tool '{name}' timed out after 30.0 seconds."
            await self._emit(
                EventType.TOOL_FAILED,
                tool_call_id=tool_call.id,
                tool_name=name,
                error=str(result),
            )
        except ToolExecutionError as e:
            result = f"Tool execution failed with error: {e}"
            await self._emit(
                EventType.TOOL_FAILED,
                tool_call_id=tool_call.id,
                tool_name=name,
                error=str(result),
            )
        except Exception as e:
            result = f"Something went wrong, tool execution failed with error: {e}"
            await self._emit(
                EventType.TOOL_FAILED,
                tool_call_id=tool_call.id,
                tool_name=name,
                error=str(result),
            )

        # The event above carries the full result; only the copy kept in the
        # conversation history is bounded.
        return Message(
            role="tool",
            content=self.agent.context.truncate_tool_output(str(result)),
            tool_call_id=tool_call.id,
        )

    async def _call_model(self) -> LLMResponse:
        """Call the LLM with a per-call timeout, retrying transient failures."""
        policy = self.agent.retry
        attempt = 0
        gen_failures = 0
        shrink_attempts = 0
        while True:
            # After a malformed-tool-call failure, nudge the model with a transient
            # hint (sent to the provider but never stored in the history).
            outgoing = self.messages
            if gen_failures:
                outgoing = [*self.messages, Message(role="user", content=GENERATION_ERROR_HINT)]
            try:
                return await asyncio.wait_for(
                    self.agent.llm.generate(
                        self.agent.model, outgoing, self.agent.tool_schemas
                    ),
                    timeout=policy.call_timeout,
                )
            except Exception as e:
                if is_request_too_large(e):
                    # Waiting can't help: the request itself exceeds the cap. Shrink old
                    # tool output to ~65% of the stated limit and re-send.
                    limit = extract_token_limit(e)
                    shrink_attempts += 1
                    fits = False
                    if limit and shrink_attempts <= 3:
                        reserved = self.agent.context.estimator.count_text(
                            json.dumps(self.agent.tool_schemas)
                        )
                        fits = shrink_to_fit(
                            self.messages,
                            int(limit * 0.65),
                            self.agent.context.estimator,
                            reserved_tokens=reserved,
                        )
                    if not fits:
                        raise LLMExecutionError(f"LLM call failed: {e}") from e
                    await self._emit(
                        EventType.MODEL_RETRY,
                        attempt=shrink_attempts,
                        delay=0.0,
                        error="request too large; trimmed older tool output",
                    )
                    continue
                if is_generation_error(e):
                    gen_failures += 1
                    if gen_failures > MAX_GENERATION_RETRIES:
                        raise LLMExecutionError(f"LLM call failed: {e}") from e
                    await self._emit(
                        EventType.MODEL_RETRY,
                        attempt=gen_failures,
                        delay=0.0,
                        error="malformed tool call; re-sampling",
                    )
                    continue
                if not is_retryable(e) or attempt + 1 >= policy.max_attempts:
                    raise LLMExecutionError(f"LLM call failed: {e}") from e
                requested = extract_retry_delay(e)
                if requested is not None and requested > policy.max_delay:
                    # Daily/hourly quota: waiting is pointless, surface it now.
                    raise LLMExecutionError(f"LLM call failed: {e}") from e
                delay = backoff_delay(attempt, policy, exc=e)
                attempt += 1
                await self._emit(
                    EventType.MODEL_RETRY,
                    attempt=attempt,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)

    MAX_ITERATIONS = 100

    async def execute(self) -> AgentResult:
        await self._emit(EventType.RUN_STARTED, task=self.user_input)
        try:
            while True:
                self.iterations += 1
                if self.iterations > self.MAX_ITERATIONS:
                    raise MaxIterationsError(
                        f"Agent exceeded {self.MAX_ITERATIONS} iterations and did not produce a result"
                    )

                compacted, info = self.agent.context.maybe_compact(
                    self.messages, self._baseline
                )
                if info is not None:
                    self.messages = compacted
                    self._baseline = None  # history changed; old count is stale
                    await self._emit(EventType.CONTEXT_COMPACTED, **info)

                await self._emit(
                    EventType.MODEL_STARTED,
                    iterations=self.iterations,
                    model=self.agent.model,
                )

                started = time.monotonic()
                sent_count = len(self.messages)
                response = await self._call_model()
                usage = response.usage
                if usage is not None:
                    self._baseline = UsageBaseline(usage.input_tokens, sent_count)

                await self._emit(
                    EventType.MODEL_COMPLETED,
                    iterations=self.iterations,
                    latency_ms=(time.monotonic() - started) * 1000,
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                )

                if response.tool_calls:
                    self.messages.append(response.to_message())
                    async with asyncio.TaskGroup() as tg:
                        tasks = [
                            tg.create_task(self._dispatch_tool(tc))
                            for tc in response.tool_calls
                        ]

                    for task in tasks:
                        self.messages.append(task.result())

                    continue
                else:
                    self.messages.append(response.to_message())
                    content = response.content or ""

                    await self._emit(
                        EventType.RUN_COMPLETED,
                        content=content,
                        iterations=self.iterations,
                        messages=self.messages,
                    )

                    return AgentResult(
                        content=content,
                        iterations=self.iterations,
                        messages=self.messages,
                    )
        except Exception as e:
            await self._emit(EventType.RUN_FAILED, error=str(e))
            raise
        finally:
            await self._event_queue.put(None)

    async def stream(self) -> AsyncIterator[AgentEvent]:
        task = asyncio.create_task(self.execute())

        try:
            while True:
                event = await self._event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                try:
                    await task
                except Exception:
                    pass
