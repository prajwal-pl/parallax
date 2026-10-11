from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from core.agent import Agent
from core.errors import LLMExecutionError
from core.llm import (
    OpenRouterClient,
    RetryPolicy,
    backoff_delay,
    is_retryable,
)
from core.types import EventType, LLMResponse, Usage

FAST = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, call_timeout=5.0)


class HTTPError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def make_agent(side_effect, retry: RetryPolicy = FAST) -> tuple[Agent, AsyncMock]:
    agent = Agent(model="m", system_prompt="s", retry=retry)
    mock = AsyncMock(side_effect=side_effect)
    agent.llm.generate = mock  # type: ignore[method-assign]
    return agent, mock


OK = LLMResponse(content="ok", tool_calls=None)


# --- is_retryable / backoff -------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_statuses(status):
    assert is_retryable(HTTPError(status))


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_non_retryable_statuses(status):
    assert not is_retryable(HTTPError(status))


def test_retryable_network_and_timeout():
    assert is_retryable(asyncio.TimeoutError())
    assert is_retryable(ConnectionError("reset"))
    assert not is_retryable(ValueError("bug"))


def test_backoff_delay_bounded_and_grows():
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0)
    for _ in range(50):
        assert 0 <= backoff_delay(0, policy) <= 1.0
        assert 0 <= backoff_delay(1, policy) <= 2.0
        assert 0 <= backoff_delay(10, policy) <= 5.0  # capped by max_delay


# --- retry flow --------------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_429_then_succeeds():
    agent, mock = make_agent([HTTPError(429), OK])
    events = [e async for e in agent.run_stream("hi")]

    assert mock.await_count == 2
    types = [e.type for e in events]
    assert types == [
        EventType.RUN_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_RETRY,
        EventType.MODEL_COMPLETED,
        EventType.RUN_COMPLETED,
    ]
    retry = events[2]
    assert retry.data["attempt"] == 1
    assert "429" in retry.data["error"]
    assert "delay" in retry.data


@pytest.mark.asyncio
async def test_run_returns_result_after_retry():
    agent, _ = make_agent([HTTPError(503), HTTPError(502), OK])
    result = await agent.run("hi")
    assert result.content == "ok"
    # retries do not count as extra iterations
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_non_retryable_fails_immediately():
    agent, mock = make_agent([HTTPError(400), OK])
    events = [e async for e in agent.run_stream("hi")]

    assert mock.await_count == 1
    assert EventType.MODEL_RETRY not in [e.type for e in events]
    assert events[-1].type == EventType.RUN_FAILED
    assert "400" in events[-1].data["error"]


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts():
    agent, mock = make_agent([HTTPError(500)] * 5)
    with pytest.raises(LLMExecutionError, match="500"):
        await agent.run("hi")

    assert mock.await_count == FAST.max_attempts


@pytest.mark.asyncio
async def test_retry_events_count_matches_attempts():
    agent, _ = make_agent([HTTPError(500)] * 5)
    events = [e async for e in agent.run_stream("hi")]

    retries = [e for e in events if e.type == EventType.MODEL_RETRY]
    assert len(retries) == FAST.max_attempts - 1
    assert [r.data["attempt"] for r in retries] == [1, 2]
    assert events[-1].type == EventType.RUN_FAILED


@pytest.mark.asyncio
async def test_timeout_is_retried():
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
        return OK

    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return await slow()
        return OK

    policy = RetryPolicy(max_attempts=3, base_delay=0, max_delay=0, call_timeout=0.05)
    agent = Agent(model="m", system_prompt="s", retry=policy)
    agent.llm.generate = flaky  # type: ignore[method-assign]

    events = [e async for e in agent.run_stream("hi")]
    assert calls["n"] == 2
    assert EventType.MODEL_RETRY in [e.type for e in events]
    assert events[-1].type == EventType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_backoff_sleep_is_awaited():
    policy = RetryPolicy(max_attempts=2, base_delay=0.5, max_delay=0.5)
    agent, _ = make_agent([HTTPError(429), OK], retry=policy)
    with patch("core.run.asyncio.sleep", new=AsyncMock()) as sleep:
        await agent.run("hi")
    sleep.assert_awaited_once()
    assert 0 <= sleep.await_args.args[0] <= 0.5  # type: ignore[union-attr]


# --- usage & latency ---------------------------------------------------------


@pytest.mark.asyncio
async def test_model_completed_includes_usage_and_latency():
    resp = LLMResponse(content="ok", usage=Usage(input_tokens=11, output_tokens=7))
    agent, _ = make_agent([resp])
    events = [e async for e in agent.run_stream("hi")]

    done = next(e for e in events if e.type == EventType.MODEL_COMPLETED)
    assert done.data["input_tokens"] == 11
    assert done.data["output_tokens"] == 7
    assert done.data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_model_completed_usage_none_when_provider_omits_it():
    agent, _ = make_agent([OK])
    events = [e async for e in agent.run_stream("hi")]
    done = next(e for e in events if e.type == EventType.MODEL_COMPLETED)
    assert done.data["input_tokens"] is None
    assert done.data["output_tokens"] is None


# --- injection ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_llm_client_is_used():
    class Scripted:
        def __init__(self):
            self.seen = []

        async def generate(self, model, messages, tools):
            self.seen.append((model, len(messages), tools))
            return LLMResponse(content="from fake")

    fake = Scripted()
    agent = Agent(model="special", system_prompt="s", llm=fake)
    result = await agent.run("hi")

    assert result.content == "from fake"
    assert fake.seen == [("special", 2, [])]


def test_openrouter_client_is_lazy(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterClient()
    assert client._client is None  # no SDK client built until first generate()


@pytest.mark.asyncio
async def test_openrouter_client_parses_response_and_usage():
    from types import SimpleNamespace as NS

    chat = NS(
        choices=[
            NS(
                message=NS(
                    content=None,
                    tool_calls=[
                        NS(
                            id="c1",
                            function=NS(name="greet", arguments='{"name": "Bob"}'),
                        ),
                        NS(id="c2", function=NS(name="greet", arguments="{bad")),
                    ],
                )
            )
        ],
        usage=NS(prompt_tokens=100, completion_tokens=20),
    )
    sdk = NS(chat=NS(send_async=AsyncMock(return_value=chat)))

    client = OpenRouterClient(api_key="k")
    client._client = sdk  # type: ignore[assignment]
    resp = await client.generate("m", [], [])

    assert resp.usage == Usage(100, 20)
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].arguments == {"name": "Bob"}
    assert resp.tool_calls[0].error is None
    assert resp.tool_calls[1].arguments == {}
    assert resp.tool_calls[1].error is not None
