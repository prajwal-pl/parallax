from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import Agent

from core.errors import (
    MaxIterationsError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from core.llm import call_llm
from core.types import AgentEvent, AgentResult, EventType, Message


class AgentRun:
    def __init__(self, agent: Agent, user_input: str):
        self.agent = agent
        self.user_input = user_input
        self.messages = [
            Message(role="system", content=agent.system_prompt),
            Message(role="user", content=user_input),
        ]
        self.iterations = 0
        self._event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def _emit(self, event_type: EventType, **data):
        """Push an event to the queue"""
        event = AgentEvent(type=event_type, data=data)
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
            if name not in self.agent.tool_registry:
                raise ToolNotFoundError(f"Tool '{name}' not registered")
            func = self.agent.tool_registry[name]
            result = await asyncio.wait_for(
                asyncio.to_thread(func, **arguments), timeout=30.0
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

        return Message(role="tool", content=str(result), tool_call_id=tool_call.id)

    MAX_ITERATIONS = 30

    async def execute(self) -> AgentResult:
        await self._emit(EventType.RUN_STARTED, task=self.user_input)
        try:
            while True:
                self.iterations += 1
                if self.iterations > self.MAX_ITERATIONS:
                    raise MaxIterationsError(
                        f"Agent exceeded {self.MAX_ITERATIONS} iterations and did not produce a result"
                    )

                await self._emit(
                    EventType.MODEL_STARTED,
                    iterations=self.iterations,
                    model=self.agent.model,
                )

                response = await call_llm(
                    model=self.agent.model,
                    messages=self.messages,
                    tools=self.agent.tool_schemas,
                )

                await self._emit(
                    EventType.MODEL_COMPLETED,
                    iterations=self.iterations,
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
