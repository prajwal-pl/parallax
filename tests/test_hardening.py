from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from parallax.core.agent import Agent
from parallax.core.errors import DuplicateToolError
from parallax.core.llm import parse_tool_arguments
from parallax.core.types import EventType, LLMResponse, ToolCall


def test_parse_tool_arguments_valid():
    assert parse_tool_arguments('{"a": 1}') == ({"a": 1}, None)


def test_parse_tool_arguments_empty():
    assert parse_tool_arguments("") == ({}, None)
    assert parse_tool_arguments(None) == ({}, None)


def test_parse_tool_arguments_invalid_json():
    args, err = parse_tool_arguments('{"a": ')
    assert args == {}
    assert err is not None and "Invalid JSON" in err


def test_parse_tool_arguments_non_object():
    args, err = parse_tool_arguments("[1, 2]")
    assert args == {}
    assert err is not None and "JSON object" in err


def test_add_tool_rejects_duplicates():
    agent = Agent(model="m", system_prompt="s")

    def greet(name: str) -> str:
        return name

    agent.add_tool(greet)
    with pytest.raises(DuplicateToolError):
        agent.add_tool(greet)
    assert len(agent.tool_schemas) == 1


@pytest.mark.asyncio
async def test_malformed_arguments_reported_to_model_not_crash():
    agent = Agent(model="m", system_prompt="s")

    def greet(name: str) -> str:
        return f"hi {name}"

    agent.add_tool(greet)

    bad = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="c1", name="greet", arguments={}, error="Invalid JSON")],
    )
    final = LLMResponse(content="recovered", tool_calls=None)

    with patch.object(agent.llm, "generate", new=AsyncMock(side_effect=[bad, final])):
        result = await agent.run("go")

    assert result.content == "recovered"
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert tool_msgs[0].content == "Error: Invalid JSON"


@pytest.mark.asyncio
async def test_events_share_run_id_and_runs_differ():
    agent = Agent(model="m", system_prompt="s")
    resp = LLMResponse(content="ok", tool_calls=None)

    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=resp)):
        first = [e async for e in agent.run_stream("a")]
        second = [e async for e in agent.run_stream("b")]

    ids_1 = {e.run_id for e in first}
    ids_2 = {e.run_id for e in second}
    assert len(ids_1) == 1 and "" not in ids_1
    assert len(ids_2) == 1
    assert ids_1 != ids_2
    assert first[-1].type == EventType.RUN_COMPLETED
