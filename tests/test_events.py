from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.agent import Agent
from core.errors import ToolError
from core.run import AgentRun
from core.types import AgentResult, EventType, LLMResponse, ToolCall


@pytest.mark.asyncio
async def test_run_stream_simple_success():
    agent = Agent(model="mock-model", system_prompt="Test System")
    mock_response = LLMResponse(content="Hello there!", tool_calls=None)

    with patch("core.run.call_llm", new=AsyncMock(return_value=mock_response)):
        events = [event async for event in agent.run_stream("Hello")]

    assert len(events) == 4
    assert events[0].type == EventType.RUN_STARTED
    assert events[0].data["task"] == "Hello"

    assert events[1].type == EventType.MODEL_STARTED
    assert events[1].data["iterations"] == 1
    assert events[1].data["model"] == "mock-model"

    assert events[2].type == EventType.MODEL_COMPLETED
    assert events[2].data["iterations"] == 1

    assert events[3].type == EventType.RUN_COMPLETED
    assert events[3].data["content"] == "Hello there!"
    assert events[3].data["iterations"] == 1

    for ev in events:
        assert isinstance(ev.timestamp, float)
        assert ev.timestamp > 0


@pytest.mark.asyncio
async def test_run_stream_with_tool_call():
    agent = Agent(model="mock-model", system_prompt="Test System")

    def greet(name: str) -> str:
        return f"Greetings, {name}!"

    agent.add_tool(greet)

    resp_1 = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="greet", arguments={"name": "Bob"})],
    )
    resp_2 = LLMResponse(content="I greeted Bob.", tool_calls=None)

    with patch("core.run.call_llm", new=AsyncMock(side_effect=[resp_1, resp_2])):
        events = [event async for event in agent.run_stream("Greet Bob")]

    event_types = [e.type for e in events]
    assert event_types == [
        EventType.RUN_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.RUN_COMPLETED,
    ]

    # Verify tool events data
    tool_started = events[3]
    assert tool_started.data["tool_name"] == "greet"
    assert tool_started.data["tool_call_id"] == "call_1"
    assert tool_started.data["arguments"] == {"name": "Bob"}

    tool_completed = events[4]
    assert tool_completed.data["tool_name"] == "greet"
    assert tool_completed.data["tool_call_id"] == "call_1"
    assert tool_completed.data["result"] == "Greetings, Bob!"

    run_completed = events[7]
    assert run_completed.data["content"] == "I greeted Bob."
    assert run_completed.data["iterations"] == 2


@pytest.mark.asyncio
async def test_run_stream_with_tool_failure():
    agent = Agent(model="mock-model", system_prompt="Test System")

    def failing_tool() -> str:
        raise ToolError("Disk full")

    agent.add_tool(failing_tool)

    resp_1 = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call_fail", name="failing_tool", arguments={})],
    )
    resp_2 = LLMResponse(content="Failed to run tool", tool_calls=None)

    with patch("core.run.call_llm", new=AsyncMock(side_effect=[resp_1, resp_2])):
        events = [event async for event in agent.run_stream("Run failing tool")]

    event_types = [e.type for e in events]
    assert EventType.TOOL_FAILED in event_types

    tool_failed = [e for e in events if e.type == EventType.TOOL_FAILED][0]
    assert tool_failed.data["tool_name"] == "failing_tool"
    assert tool_failed.data["tool_call_id"] == "call_fail"
    assert "Disk full" in tool_failed.data["error"]


@pytest.mark.asyncio
async def test_run_stream_parallel_tools():
    agent = Agent(model="mock-model", system_prompt="Test System")

    def tool_a(x: int) -> int:
        return x * 2

    def tool_b(y: str) -> str:
        return y.upper()

    agent.add_tool(tool_a)
    agent.add_tool(tool_b)

    resp_1 = LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id="call_a", name="tool_a", arguments={"x": 5}),
            ToolCall(id="call_b", name="tool_b", arguments={"y": "hello"}),
        ],
    )
    resp_2 = LLMResponse(content="Done both", tool_calls=None)

    with patch("core.run.call_llm", new=AsyncMock(side_effect=[resp_1, resp_2])):
        events = [event async for event in agent.run_stream("Parallel test")]

    started_tools = {
        e.data["tool_name"] for e in events if e.type == EventType.TOOL_STARTED
    }
    completed_tools = {
        e.data["tool_name"] for e in events if e.type == EventType.TOOL_COMPLETED
    }
    assert started_tools == {"tool_a", "tool_b"}
    assert completed_tools == {"tool_a", "tool_b"}

    run_completed = [e for e in events if e.type == EventType.RUN_COMPLETED][0]
    assert run_completed.data["content"] == "Done both"


@pytest.mark.asyncio
async def test_run_stream_max_iterations_failed(monkeypatch):
    agent = Agent(model="mock-model", system_prompt="Test System")

    def noop() -> str:
        return "noop"

    agent.add_tool(noop)

    monkeypatch.setattr(AgentRun, "MAX_ITERATIONS", 2)

    infinite_resp = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call_loop", name="noop", arguments={})],
    )

    with patch("core.run.call_llm", new=AsyncMock(return_value=infinite_resp)):
        events = [event async for event in agent.run_stream("Loop forever")]

    last_event = events[-1]
    assert last_event.type == EventType.RUN_FAILED
    assert "exceeded 2 iterations" in last_event.data["error"]


@pytest.mark.asyncio
async def test_run_stream_llm_exception():
    agent = Agent(model="mock-model", system_prompt="Test System")

    with patch(
        "core.run.call_llm",
        new=AsyncMock(side_effect=RuntimeError("API Network Down")),
    ):
        events = [event async for event in agent.run_stream("Network issue")]

    event_types = [e.type for e in events]
    assert event_types == [
        EventType.RUN_STARTED,
        EventType.MODEL_STARTED,
        EventType.RUN_FAILED,
    ]
    assert "API Network Down" in events[-1].data["error"]


@pytest.mark.asyncio
async def test_run_stream_early_break():
    agent = Agent(model="mock-model", system_prompt="Test System")
    mock_resp = LLMResponse(content="Answer", tool_calls=None)

    collected = []
    with patch("core.run.call_llm", new=AsyncMock(return_value=mock_resp)):
        async for event in agent.run_stream("Task"):
            collected.append(event)
            break

    assert len(collected) == 1
    assert collected[0].type == EventType.RUN_STARTED


@pytest.mark.asyncio
async def test_agent_run_backward_compatibility():
    agent = Agent(model="mock-model", system_prompt="Test System")
    mock_resp = LLMResponse(content="Final result text", tool_calls=None)

    with patch("core.run.call_llm", new=AsyncMock(return_value=mock_resp)):
        result = await agent.run("Perform run")

    assert isinstance(result, AgentResult)
    assert result.content == "Final result text"
    assert result.iterations == 1
    assert len(result.messages) == 3
