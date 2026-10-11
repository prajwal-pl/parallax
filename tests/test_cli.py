from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from rich.console import Console
from typer.testing import CliRunner

from parallax.cli.app import app, build_default_agent
from parallax.cli.doctor import run_doctor
from parallax.cli.renderer import CliRenderer
from parallax.cli.repl import InteractiveRepl
from parallax.cli.theme import PARALLAX_THEME
from parallax.core.types import AgentEvent, EventType, Message

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "parallax" in result.stdout
    assert "v0.1.0" in result.stdout


def test_cli_tools():
    result = runner.invoke(app, ["tools"])
    assert result.exit_code == 0
    assert "bash" in result.stdout
    assert "str_replace_editor" in result.stdout
    assert "search_code" in result.stdout


def test_cli_doctor(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test12345678")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    console = Console(record=True, theme=PARALLAX_THEME)
    healthy = run_doctor(console=console)
    output = console.export_text()

    assert healthy is True
    assert "Python" in output
    assert "OPENROUTER_API_KEY" in output
    assert "Git Repository" in output


def test_cli_doctor_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    console = Console(record=True, theme=PARALLAX_THEME)
    healthy = run_doctor(console=console)
    output = console.export_text()

    assert healthy is False
    assert "missing" in output


def test_build_default_agent():
    agent = build_default_agent(model="test-model", workspace=".")
    assert agent.model == "test-model"
    assert "bash" in agent.tool_registry
    assert "str_replace_editor" in agent.tool_registry
    assert "search_code" in agent.tool_registry


def test_renderer_event_stream():
    console = Console(record=True, theme=PARALLAX_THEME)
    renderer = CliRenderer(console=console)

    # 1. Run started
    renderer.handle_event(AgentEvent(type=EventType.RUN_STARTED, data={"task": "Fix tests"}))
    # 2. Model started
    renderer.handle_event(AgentEvent(type=EventType.MODEL_STARTED, data={"iteration": 1, "model": "test-model"}))
    # 3. Model retry
    renderer.handle_event(AgentEvent(type=EventType.MODEL_RETRY, data={"attempt": 1, "delay": 1.0, "error": "Rate limit"}))
    # 4. Model completed
    renderer.handle_event(AgentEvent(type=EventType.MODEL_COMPLETED, data={"iteration": 1, "tool_calls_count": 1, "content": "Running bash"}))
    # 5. Tool started
    renderer.handle_event(AgentEvent(type=EventType.TOOL_STARTED, data={"tool_call_id": "tc1", "tool_name": "bash", "arguments": {"command": "pytest"}}))
    # 6. Tool completed
    renderer.handle_event(AgentEvent(type=EventType.TOOL_COMPLETED, data={"tool_call_id": "tc1", "tool_name": "bash", "result": "1 passed"}))
    # 7. Context compacted
    renderer.handle_event(AgentEvent(type=EventType.CONTEXT_COMPACTED, data={"tokens_before": 50000, "tokens_after": 10000, "dropped_turns": 4}))
    # 8. Tool failed
    renderer.handle_event(AgentEvent(type=EventType.TOOL_FAILED, data={"tool_call_id": "tc2", "tool_name": "editor", "error": "File not found"}))
    # 9. Run completed
    renderer.handle_event(AgentEvent(type=EventType.RUN_COMPLETED, data={"iterations": 1, "content": "Everything is done!"}))

    output = console.export_text()

    assert "Fix tests" in output
    assert "bash" in output
    assert "1 passed" in output
    assert "context compacted" in output
    assert "Everything is done!" in output
    assert "completed in" in output


def test_renderer_run_failed():
    console = Console(record=True, theme=PARALLAX_THEME)
    renderer = CliRenderer(console=console)

    renderer.handle_event(AgentEvent(type=EventType.RUN_STARTED, data={"task": "Broken task"}))
    renderer.handle_event(AgentEvent(type=EventType.RUN_FAILED, data={"error": "Fatal provider failure"}))

    output = console.export_text()
    assert "run failed" in output
    assert "Fatal provider failure" in output


@pytest.mark.asyncio
async def test_repl_slash_commands():
    agent = build_default_agent()
    console = Console(record=True, theme=PARALLAX_THEME)
    repl = InteractiveRepl(agent=agent, console=console)

    # Test /help
    assert await repl._handle_command("/help") is None
    # Test /model
    assert await repl._handle_command("/model") is None
    assert await repl._handle_command("/model new/custom-model") is None
    assert agent.model == "new/custom-model"
    # Test /context
    assert await repl._handle_command("/context") is None
    # Test /tools
    assert await repl._handle_command("/tools") is None
    # Test /compact (empty history)
    assert await repl._handle_command("/compact") is None
    # Test /clear
    repl.messages = [Message(role="user", content="hello")]
    assert await repl._handle_command("/clear") is None
    assert len(repl.messages) == 0
    # Test /exit
    assert await repl._handle_command("/exit") == "exit"
    assert await repl._handle_command("/quit") == "exit"

    output = console.export_text()
    assert "switched model from" in output
    assert "conversation history cleared" in output
