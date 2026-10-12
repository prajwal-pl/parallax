# import ast
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

import groq
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
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 20.0
    call_timeout: float = 120.0


def is_retryable(exc: BaseException) -> bool:
    """Transient failures worth retrying: timeouts, network drops, 408/429/5xx, rate limits."""
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, httpx.TransportError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status in (408, 429) or 500 <= status < 600):
        return True
    msg = str(exc).lower()
    if any(k in msg for k in ("rate limit", "rate_limit", "rate_limit_exceeded", "429", "too many requests", "tokens per minute", "tpm", "rpm")):
        return True
    return False


_GENERATION_ERROR_MARKERS = (
    "tool call validation failed",
    "output_parse_failed",
    "tool_use_failed",
    "failed_generation",
    "parsing failed",
    "failed to call a function",
    "which was not in request.tools",
)


def is_generation_error(exc: BaseException) -> bool:
    """The model emitted a malformed/invalid tool call (provider 400). Re-sampling usually fixes it."""
    msg = str(exc).lower()
    return any(k in msg for k in _GENERATION_ERROR_MARKERS)


GENERATION_ERROR_HINT = (
    "Your previous tool call was rejected as malformed. Retry with a valid call: use only the "
    "declared tool names, provide every required argument with the correct JSON type, and omit "
    "optional arguments instead of passing null."
)


def extract_retry_delay(exc: BaseException) -> float | None:
    """Extract requested retry delay from HTTP response headers or error messages."""
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        for k in ("retry-after", "Retry-After"):
            retry_after = headers.get(k)
            if retry_after:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
        for k in (
            "x-ratelimit-reset-tokens",
            "X-RateLimit-Reset-Tokens",
            "x-ratelimit-reset-requests",
            "X-RateLimit-Reset-Requests",
        ):
            reset_tokens = headers.get(k)
            if reset_tokens and isinstance(reset_tokens, str):
                clean = reset_tokens.rstrip("s").strip()
                try:
                    return float(clean)
                except (ValueError, TypeError):
                    pass

    msg = str(exc)
    match = re.search(r"try again in\s+((?:[\d.]+(?:ms|h|m|s)\s*)+)", msg, re.IGNORECASE)
    if match:
        total = 0.0
        found = False
        for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", match.group(1), re.IGNORECASE):
            try:
                n = float(value)
            except ValueError:
                continue
            found = True
            total += n * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit.lower()]
        if found:
            return total

    return None


def backoff_delay(attempt: int, policy: RetryPolicy, exc: BaseException | None = None) -> float:
    """Exponential backoff with full jitter, respecting provider Retry-After if available."""
    if exc is not None:
        explicit = extract_retry_delay(exc)
        if explicit is not None and explicit > 0:
            return min(policy.max_delay, explicit + 0.3)

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

    async def list_models(self) -> list[str]:
        """Fetch available models from OpenRouter."""
        return await fetch_openrouter_models(api_key=self._api_key)


class GroqClient:
    def __init__(self, api_key: str | None = None):
        if groq is None:
            raise ImportError("groq package is not installed")
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client: groq.Groq | None = None

    async def generate(
        self, model: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        if self._client is None:
            self._client = groq.Groq(api_key=self._api_key)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": cast(Any, [m.to_dict() for m in messages]),
        }
        if tools:
            kwargs["tools"] = cast(Any, tools)
            kwargs["tool_choice"] = "auto"

        # Groq SDK is synchronous; run in a thread to keep the API async
        chat = await asyncio.to_thread(
            self._client.chat.completions.create,
            **kwargs,
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

    async def list_models(self) -> list[str]:
        """Fetch available models from Groq."""
        return await fetch_groq_models(client=self._client, api_key=self._api_key)


async def fetch_openrouter_models(api_key: str | None = None) -> list[str]:
    """Fetch live list of available model IDs from OpenRouter API."""
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.get("https://openrouter.ai/api/v1/models", headers=headers)
        r.raise_for_status()
        data = r.json().get("data", [])
        return [m["id"] for m in data if isinstance(m, dict) and "id" in m]


async def fetch_groq_models(
    client: groq.Groq | None = None, api_key: str | None = None
) -> list[str]:
    """Fetch live list of available model IDs from Groq API, filtering for active chat models."""
    if groq is None:
        raise ImportError("groq package is not installed")
    if client is None:
        key = api_key or os.getenv("GROQ_API_KEY")
        client = groq.Groq(api_key=key)
    models = await asyncio.to_thread(client.models.list)

    EXCLUDED_SUBSTRINGS = (
        "whisper",
        "prompt-guard",
        "orpheus",
        "safeguard",
        "allam",  # no tool-calling support
    )
    result: list[str] = []
    for m in models.data:
        mid = getattr(m, "id", None)
        if not mid or not isinstance(mid, str):
            continue
        mid_lower = mid.lower()
        if any(exc in mid_lower for exc in EXCLUDED_SUBSTRINGS):
            continue
        ctx = getattr(m, "context_window", None)
        if isinstance(ctx, int) and ctx < 1024:
            continue
        if getattr(m, "active", True) is False:
            continue
        result.append(mid)

    PREFERRED_ORDER = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    sorted_result: list[str] = []
    for pref in PREFERRED_ORDER:
        if pref in result:
            sorted_result.append(pref)
    for m in result:
        if m not in sorted_result:
            sorted_result.append(m)

    return sorted_result if sorted_result else result


async def call_llm(model: str, messages: list, tools: list) -> LLMResponse:
    """
    Call the LLM with the given model, messages, and tools.
    Thin wrapper kept for backwards compatibility.
    """
    return await OpenRouterClient().generate(model, messages, tools)
