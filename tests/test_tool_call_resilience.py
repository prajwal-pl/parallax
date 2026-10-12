from __future__ import annotations

import pytest

from parallax.core.agent import Agent
from parallax.core.errors import LLMExecutionError
from parallax.core.llm import is_generation_error
from parallax.core.tool import extract_schema
from parallax.core.types import EventType, LLMResponse, Message, ToolCall


class FlakyLLM:
    """Fails with a Groq-style malformed tool call N times, then succeeds."""

    def __init__(self, failures: int, msg: str = "Error code: 400 - Tool call validation failed: parameters did not match schema"):
        self.failures = failures
        self.msg = msg
        self.calls: list[list[Message]] = []

    async def generate(self, model, messages, tools):
        self.calls.append(list(messages))
        if len(self.calls) <= self.failures:
            raise RuntimeError(self.msg)
        return LLMResponse(content="done", tool_calls=None, usage=None)


def test_is_generation_error():
    assert is_generation_error(RuntimeError("Tool call validation failed: x"))
    assert is_generation_error(RuntimeError("{'code': 'output_parse_failed'}"))
    assert not is_generation_error(RuntimeError("401 unauthorized"))


@pytest.mark.asyncio
async def test_malformed_tool_call_is_resampled():
    llm = FlakyLLM(failures=2)
    agent = Agent(model="m", system_prompt="s", llm=llm)
    result = await agent.run("hi")
    assert result.content == "done"
    assert len(llm.calls) == 3
    assert llm.calls[0][-1].content == "hi"
    assert "malformed" in llm.calls[1][-1].content  # transient hint
    # hint is never persisted in the history
    assert all("malformed" not in (m.content or "") for m in result.messages)


@pytest.mark.asyncio
async def test_malformed_tool_call_gives_up():
    llm = FlakyLLM(failures=99)
    agent = Agent(model="m", system_prompt="s", llm=llm)
    with pytest.raises(LLMExecutionError):
        await agent.run("hi")
    assert len(llm.calls) == 5  # 1 + 4 resamples


@pytest.mark.asyncio
async def test_null_optional_args_are_dropped():
    seen = {}

    def probe(path: str, view_range: list[int] | None = None) -> str:
        """Probe.

        Args:
            path: a path
            view_range: optional range
        """
        seen.update(path=path, view_range=view_range)
        return "ok"

    class L:
        n = 0

        async def generate(self, model, messages, tools):
            self.n += 1
            if self.n == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="1", name="probe", arguments={"path": "a", "view_range": None})],
                    usage=None,
                )
            return LLMResponse(content="fin", tool_calls=None, usage=None)

    agent = Agent(model="m", system_prompt="s", llm=L())
    agent.add_tool(probe)
    result = await agent.run("go")
    assert result.content == "fin"
    assert seen == {"path": "a", "view_range": None}  # default used, no TypeError


def test_schema_has_no_null_defaults():
    def f(a: str, b: str | None = None, c: int = 3) -> str:
        return a

    props = extract_schema(f).to_dict()["function"]["parameters"]["properties"]
    assert "default" not in props["b"]
    assert props["c"]["default"] == 3
