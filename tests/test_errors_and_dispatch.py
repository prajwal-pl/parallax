import asyncio
import time
import pytest
from core.agent import Agent
from core.run import AgentRun
from core.errors import ToolError
from core.types import ToolCall


@pytest.mark.asyncio
async def test_dispatch_tool_success():
    agent = Agent(model="mock-model", system_prompt="Test")
    
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    agent.add_tool(greet)
    run = AgentRun(agent, "Hi")

    tc = ToolCall(id="call_1", name="greet", arguments={"name": "Alice"})
    msg = await run._dispatch_tool(tc)

    assert msg.role == "tool"
    assert msg.content == "Hello, Alice!"
    assert msg.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_dispatch_tool_raises_tool_error():
    agent = Agent(model="mock-model", system_prompt="Test")

    def failing_tool(path: str) -> str:
        raise ToolError(f"File not found: {path}")

    agent.add_tool(failing_tool)
    run = AgentRun(agent, "Hi")

    tc = ToolCall(id="call_2", name="failing_tool", arguments={"path": "missing.txt"})
    msg = await run._dispatch_tool(tc)

    assert msg.role == "tool"
    assert msg.content == "Error: File not found: missing.txt"
    assert msg.tool_call_id == "call_2"


@pytest.mark.asyncio
async def test_dispatch_tool_not_found():
    agent = Agent(model="mock-model", system_prompt="Test")
    run = AgentRun(agent, "Hi")

    tc = ToolCall(id="call_3", name="non_existent", arguments={})
    msg = await run._dispatch_tool(tc)

    assert msg.role == "tool"
    assert "Error: Tool 'non_existent' not registered" in msg.content
    assert msg.tool_call_id == "call_3"


@pytest.mark.asyncio
async def test_dispatch_tool_unexpected_exception():
    agent = Agent(model="mock-model", system_prompt="Test")

    def buggy_tool() -> str:
        return str(1 / 0)

    agent.add_tool(buggy_tool)
    run = AgentRun(agent, "Hi")

    tc = ToolCall(id="call_4", name="buggy_tool", arguments={})
    msg = await run._dispatch_tool(tc)

    assert msg.role == "tool"
    assert "Something went wrong" in msg.content
    assert "division by zero" in msg.content
    assert msg.tool_call_id == "call_4"


@pytest.mark.asyncio
async def test_dispatch_tool_timeout():
    from unittest.mock import patch

    agent = Agent(model="mock-model", system_prompt="Test")

    def slow_tool() -> str:
        return "done"

    agent.add_tool(slow_tool)
    run = AgentRun(agent, "Hi")

    tc = ToolCall(id="call_5", name="slow_tool", arguments={})

    async def fake_wait_for(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.wait_for", side_effect=fake_wait_for):
        msg = await run._dispatch_tool(tc)

    assert msg.role == "tool"
    assert "timed out after 30.0 seconds" in msg.content
    assert msg.tool_call_id == "call_5"
