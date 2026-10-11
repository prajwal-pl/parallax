# import ast
import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx
from dotenv import load_dotenv
from openrouter import OpenRouter

from parallax.core.types import LLMResponse, Message, ToolCall, Usage

load_dotenv()


# def parse_arguments(raw: str) -> dict:
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         try:
#             return ast.literal_eval(raw)
#         except Exception:
#             return {"raw": raw}


def parse_tool_arguments(raw: str | None) -> tuple[dict, str | None]:
    """
    Parse raw JSON tool arguments from the model.
    Returns (arguments, error). On failure arguments is {} and error describes why,
    so the model can be told and retry rather than crashing the run.
    """
    if raw is None or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON in tool arguments ({e}). Raw: {raw[:200]}"
    if not isinstance(parsed, dict):
        return {}, f"Tool arguments must be a JSON object, got {type(parsed).__name__}."
    return parsed, None


class LLMClient(Protocol):
    """Anything that can turn a conversation into an LLMResponse."""

    async def generate(
        self, model: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 20.0
    call_timeout: float = 120.0


def is_retryable(exc: BaseException) -> bool:
    """Transient failures worth retrying: timeouts, network drops, 408/429/5xx."""
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, httpx.TransportError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in (408, 429) or 500 <= status < 600)


def backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    """Exponential backoff with full jitter. `attempt` is 0-based."""
    ceiling = min(policy.max_delay, policy.base_delay * 2**attempt)
    return random.uniform(0, ceiling)


class OpenRouterClient:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._client: OpenRouter | None = None  # created lazily, not at import time

    async def generate(
        self, model: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        if self._client is None:
            self._client = OpenRouter(api_key=self._api_key)

        chat = await self._client.chat.send_async(
            model=model,
            # The SDK's typed unions don't accept plain dicts; the wire format is identical.
            messages=cast(Any, [m.to_dict() for m in messages]),
            tools=cast(Any, tools),
            tool_choice="auto",
        )

        response = chat.choices[0].message
        chat_response = response.content
        calls = response.tool_calls

        tool_calls: list[ToolCall] | None = None
        if calls:
            tool_calls = []
            for tc in calls:
                arguments, error = parse_tool_arguments(tc.function.arguments)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                        error=error,
                    )
                )

        usage: Usage | None = None
        raw_usage = getattr(chat, "usage", None)
        if raw_usage is not None:
            usage = Usage(
                input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            )

        return LLMResponse(
            content=chat_response if isinstance(chat_response, str) else None,
            tool_calls=tool_calls,
            usage=usage,
        )


async def call_llm(model: str, messages: list, tools: list) -> LLMResponse:
    """
    Call the LLM with the given model, messages, and tools.
    Thin wrapper kept for backwards compatibility.
    """
    return await OpenRouterClient().generate(model, messages, tools)
