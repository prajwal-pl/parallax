from __future__ import annotations

import pytest

from parallax.core.agent import Agent
from parallax.core.context.shrink import (
    extract_token_limit,
    is_request_too_large,
    shrink_to_fit,
)
from parallax.core.context.tokens import TokenEstimator
from parallax.core.errors import LLMExecutionError
from parallax.core.types import LLMResponse, Message, ToolCall

ERR_413 = (
    "Error code: 413 - {'error': {'message': 'Request too large for model `x` ... on tokens "
    "per minute (TPM): Limit 8000, Requested 8756, please reduce your message size', "
    "'code': 'rate_limit_exceeded'}}"
)


def test_detects_and_parses_413():
    e = RuntimeError(ERR_413)
    assert is_request_too_large(e)
    assert extract_token_limit(e) == 8000
    assert not is_request_too_large(RuntimeError("429 rate limit, try again in 2s"))


def test_shrink_clips_old_tool_output_only():
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="question"),
        Message(role="tool", content="A" * 20000, tool_call_id="1"),
        Message(role="tool", content="B" * 20000, tool_call_id="2"),
    ]
    est = TokenEstimator()
    assert est.count_messages(msgs) > 5000
    assert shrink_to_fit(msgs, 1500, est)
    assert est.count_messages(msgs) <= 1500
    assert msgs[0].content == "sys" and msgs[1].content == "question"
    assert "omitted" in msgs[2].content


def test_shrink_reports_failure_when_unfittable():
    msgs = [Message(role="user", content="word " * 20000)]
    assert shrink_to_fit(msgs, 1000, TokenEstimator()) is False


class BigThenOK:
    def __init__(self):
        self.sizes: list[int] = []

    async def generate(self, model, messages, tools):
        self.sizes.append(sum(len(m.content) for m in messages))
        if len(self.sizes) == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="big", arguments={})],
                usage=None,
            )
        if self.sizes[-1] > 20000:
            raise RuntimeError(ERR_413)
        return LLMResponse(content="ok", tool_calls=None, usage=None)


@pytest.mark.asyncio
async def test_run_recovers_from_413_by_trimming():
    def big() -> str:
        """Return a huge string."""
        return "x" * 60000

    llm = BigThenOK()
    agent = Agent(model="m", system_prompt="s", llm=llm)
    agent.context.config.max_tool_chars = 100_000  # keep the output huge in history
    agent.add_tool(big)
    result = await agent.run("go")
    assert result.content == "ok"
    assert llm.sizes[-1] < llm.sizes[-2]


@pytest.mark.asyncio
async def test_413_without_limit_fails_fast():
    class L:
        n = 0

        async def generate(self, model, messages, tools):
            self.n += 1
            raise RuntimeError("Error code: 413 - payload too large")

    llm = L()
    agent = Agent(model="m", system_prompt="s", llm=llm)
    with pytest.raises(LLMExecutionError):
        await agent.run("hi")
    assert llm.n == 1


def test_extract_retry_delay_compound_durations():
    from parallax.core.llm import extract_retry_delay

    assert extract_retry_delay(RuntimeError("Please try again in 27m29.808s.")) == pytest.approx(1649.808)
    assert extract_retry_delay(RuntimeError("try again in 2.37s")) == pytest.approx(2.37)
    assert extract_retry_delay(RuntimeError("try again in 250ms")) == pytest.approx(0.25)
    assert extract_retry_delay(RuntimeError("try again in 1h2m")) == pytest.approx(3720)


@pytest.mark.asyncio
async def test_daily_quota_fails_fast_without_sleeping():
    class L:
        n = 0

        async def generate(self, model, messages, tools):
            self.n += 1
            raise RuntimeError("Error code: 429 - Rate limit reached on tokens per day (TPD). Please try again in 27m29.8s.")

    llm = L()
    agent = Agent(model="m", system_prompt="s", llm=llm)
    with pytest.raises(LLMExecutionError):
        await agent.run("hi")
    assert llm.n == 1
