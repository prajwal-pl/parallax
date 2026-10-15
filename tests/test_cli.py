from __future__ import annotations

import io
import os
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
    assert "v0.1.5" in result.stdout


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
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    console = Console(record=True, theme=PARALLAX_THEME)
    healthy = run_doctor(console=console)
    output = console.export_text()

    assert healthy is False
    assert "unset" in output


def test_build_default_agent():
    agent = build_default_agent(model="test-model", workspace=".")
    assert agent.model == "test-model"
    assert "bash" in agent.tool_registry
    assert "str_replace_editor" in agent.tool_registry
    assert "search_code" in agent.tool_registry


def test_get_default_model_auto_selects_groq(monkeypatch):
    from parallax.cli.app import get_default_model
    monkeypatch.delenv("PARALLAX_MODEL", raising=False)
    monkeypatch.delenv("PARALLAX_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    assert get_default_model() == "llama-3.3-70b-versatile"


def test_get_default_model_with_openrouter(monkeypatch):
    from parallax.cli.app import get_default_model
    monkeypatch.delenv("PARALLAX_MODEL", raising=False)
    monkeypatch.delenv("PARALLAX_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    assert get_default_model() == "liquid/lfm-2.5-2.6b:free"


def test_global_config_load_and_save(tmp_path, monkeypatch):
    from parallax.cli.config import get_config_file, load_global_config, save_global_config
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    saved_path = save_global_config({"GROQ_API_KEY": "gsk_saved_key"})
    assert saved_path == tmp_path / ".parallax" / "config.json"
    assert saved_path.is_file()

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    loaded = load_global_config()
    assert loaded.get("GROQ_API_KEY") == "gsk_saved_key"
    assert os.environ.get("GROQ_API_KEY") == "gsk_saved_key"


def test_prompt_onboarding_non_interactive(monkeypatch):
    from parallax.cli.config import prompt_onboarding
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert prompt_onboarding() is False


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
    assert "Execution Failed" in output
    assert "Suggestions" in output


def test_renderer_abstracted_rate_limit():
    console = Console(record=True, theme=PARALLAX_THEME)
    renderer = CliRenderer(console=console)

    renderer.handle_event(AgentEvent(type=EventType.RUN_FAILED, data={"error": "Rate limit exceeded: free-models-per-day"}))

    output = console.export_text()
    assert "Provider Quota Limit" in output
    assert "Switch model" in output
    assert "/model" in output


def test_renderer_interactive_no_duplicate_echo():
    console = Console(record=True, theme=PARALLAX_THEME)
    renderer = CliRenderer(console=console, is_interactive=True)

    renderer.handle_event(AgentEvent(type=EventType.RUN_STARTED, data={"task": "test prompt"}))
    output = console.export_text()
    # In interactive mode, task should not be echoed again
    assert "» test prompt" not in output


@pytest.mark.asyncio
async def test_repl_slash_commands(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test12345678")
    agent = build_default_agent()
    console = Console(record=True, theme=PARALLAX_THEME)
    repl = InteractiveRepl(agent=agent, console=console)

    # Test /help
    assert await repl._handle_command("/help") is None
    # Test /model
    assert await repl._handle_command("/model") is None
    assert await repl._handle_command("/model new/custom-model") is None
    assert agent.model == "new/custom-model"
    output_before_clear = console.export_text()
    assert "switched model from" in output_before_clear

    # Test /context
    assert await repl._handle_command("/context") is None
    # Test /tools
    assert await repl._handle_command("/tools") is None
    # Test /doctor
    assert await repl._handle_command("/doctor") is None
    # Test /compact (empty history)
    assert await repl._handle_command("/compact") is None

    # Test aliases
    assert await repl._handle_command("/q") == "exit"
    assert await repl._handle_command("/h") is None
    assert await repl._handle_command("/c") is None
    assert await repl._handle_command("/t") is None
    assert await repl._handle_command("/d") is None

    # Test /cls (which clears the console and re-renders the hero banner)
    repl.messages = [Message(role="user", content="hello")]
    assert await repl._handle_command("/cls") is None
    assert len(repl.messages) == 0

    output_after_clear = console.export_text()
    assert "PARALLAX" in output_after_clear

    # Test /help topic filtering and ? alias
    console_help = Console(record=True, theme=PARALLAX_THEME)
    repl_help = InteractiveRepl(agent=agent, console=console_help)

    assert await repl_help._handle_command("/help") is None
    out_all = console_help.export_text()
    assert "parallax guide" in out_all
    assert "Commands" in out_all
    assert "Keyboard Shortcuts" in out_all

    console_cmd = Console(record=True, theme=PARALLAX_THEME)
    repl_cmd = InteractiveRepl(agent=agent, console=console_cmd)
    assert await repl_cmd._handle_command("/help commands") is None
    out_cmd = console_cmd.export_text()
    assert "command reference" in out_cmd
    assert "/model" in out_cmd

    console_keys = Console(record=True, theme=PARALLAX_THEME)
    repl_keys = InteractiveRepl(agent=agent, console=console_keys)
    assert await repl_keys._handle_command("/help shortcuts") is None
    out_keys = console_keys.export_text()
    assert "keyboard shortcuts" in out_keys
    assert "Ctrl + C" in out_keys

    console_err = Console(record=True, theme=PARALLAX_THEME)
    repl_err = InteractiveRepl(agent=agent, console=console_err)
    assert await repl_err._handle_command("/help unknown") is None
    out_err = console_err.export_text()
    assert "unknown help topic 'unknown'" in out_err

    assert await repl_help._handle_command("?") is None


def test_slash_command_completer():
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document
    from parallax.cli.repl import SlashCommandCompleter

    completer = SlashCommandCompleter()
    doc_m = Document(text="/m", cursor_position=2)
    completions_m = [c.text for c in completer.get_completions(doc_m, CompleteEvent(text_inserted=False))]
    assert "/model" in completions_m

    doc_h = Document(text="/h", cursor_position=2)
    completions_h = [c.text for c in completer.get_completions(doc_h, CompleteEvent(text_inserted=False))]
    assert "/help" in completions_h
    assert "/help commands" in completions_h
    assert "/help shortcuts" in completions_h

    # Test /model <space> completion
    doc_model_space = Document(text="/model ", cursor_position=7)
    completions_models = [c.text for c in completer.get_completions(doc_model_space, CompleteEvent(text_inserted=False))]
    assert "liquid/lfm-2.5-2.6b:free" in completions_models
    assert "openai/gpt-4o-mini" in completions_models
    assert "llama-3.3-70b-versatile" in completions_models

    # Test /m <space> prefix filter
    doc_m_filter = Document(text="/m open", cursor_position=6)
    completions_filter = [c.text for c in completer.get_completions(doc_m_filter, CompleteEvent(text_inserted=False))]
    assert "openai/gpt-4o-mini" in completions_filter
    assert "llama-3.3-70b-versatile" not in completions_filter


@pytest.mark.asyncio
async def test_repl_pick_model_interactive():
    agent = build_default_agent()
    console = Console(record=True, theme=PARALLAX_THEME)
    repl = InteractiveRepl(agent=agent, console=console)

    # 1. Test selecting an OpenRouter model via picker
    with patch("sys.stdin.isatty", return_value=True), patch(
        "prompt_toolkit.application.Application.run_async", new=AsyncMock(return_value="openai/gpt-4o-mini")
    ):
        await repl._handle_command("/model")
        assert agent.model == "openai/gpt-4o-mini"
        assert agent.provider_name == "OpenRouter"

    # 2. Test selecting a Groq model via picker
    with patch("sys.stdin.isatty", return_value=True), patch(
        "prompt_toolkit.application.Application.run_async", new=AsyncMock(return_value="llama-3.3-70b-versatile")
    ):
        await repl._handle_command("/model")
        assert agent.model == "llama-3.3-70b-versatile"
        assert agent.provider_name == "Groq"

    # 3. Test selecting custom model
    mock_prompt_session = AsyncMock()
    mock_prompt_session.prompt_async.return_value = "custom-provider/special-model"
    with patch("sys.stdin.isatty", return_value=True), patch(
        "prompt_toolkit.application.Application.run_async", new=AsyncMock(return_value="__custom__")
    ), patch.object(repl, "_get_prompt_session", return_value=mock_prompt_session):
        await repl._handle_command("/model")
        assert agent.model == "custom-provider/special-model"

    # 4. Test cancelling picker
    with patch("sys.stdin.isatty", return_value=True), patch(
        "prompt_toolkit.application.Application.run_async", new=AsyncMock(return_value=None)
    ):
        await repl._handle_command("/model")
        assert agent.model == "custom-provider/special-model"
        out = console.export_text()
        assert "selection cancelled" in out


@pytest.mark.asyncio
async def test_agent_set_model_switching():
    from parallax.core.llm import GroqClient, OpenRouterClient
    agent = build_default_agent(model="liquid/lfm-2.5-2.6b:free")
    assert isinstance(agent.llm, OpenRouterClient)

    # Switch to groq prefix
    agent.set_model("groq/llama-3.1-8b-instant")
    assert agent.model == "llama-3.1-8b-instant"
    assert isinstance(agent.llm, GroqClient)
    assert agent.provider_name == "Groq"

    # Switch to standard groq model without prefix
    agent.set_model("llama-3.3-70b-versatile")
    assert agent.model == "llama-3.3-70b-versatile"
    assert isinstance(agent.llm, GroqClient)

    # Switch back to openrouter model
    agent.set_model("anthropic/claude-3.5-sonnet")
    assert agent.model == "anthropic/claude-3.5-sonnet"
    assert isinstance(agent.llm, OpenRouterClient)
    assert agent.provider_name == "OpenRouter"


@pytest.mark.asyncio
async def test_repl_run_async_prompt():
    import warnings
    agent = build_default_agent()
    console = Console(record=True, theme=PARALLAX_THEME)
    repl = InteractiveRepl(agent=agent, console=console)

    # Mock session prompt_async to return /q to test clean shutdown without warnings
    mock_session = AsyncMock()
    mock_session.prompt_async.return_value = "/q"

    with patch("sys.stdin.isatty", return_value=True), patch.object(
        repl, "_get_prompt_session", return_value=mock_session
    ):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            await repl.run()

            # Ensure no RuntimeWarning regarding unawaited Application.run_async
            runtime_warnings = [
                w for w in recorded_warnings
                if issubclass(w.category, RuntimeWarning) and "run_async" in str(w.message)
            ]
            assert len(runtime_warnings) == 0

    output = console.export_text()
    assert "PARALLAX" in output
    assert "session ended" in output


def test_onboarding_configure_both_providers(tmp_path, monkeypatch):
    from parallax.cli.config import prompt_onboarding, get_config_file
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    console = Console(record=True, theme=PARALLAX_THEME)
    inputs = iter(["3", "gsk_testgroq123", "sk-or-testrouter456"])
    monkeypatch.setattr(console, "input", lambda *args: next(inputs))

    res = prompt_onboarding(console=console)
    assert res is True
    assert os.environ.get("GROQ_API_KEY") == "gsk_testgroq123"
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-testrouter456"

    out = console.export_text()
    assert "Both providers configured" in out


def test_cli_config_and_key_commands(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1234567890")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Test 'parallax config' (show status)
    result_show = runner.invoke(app, ["config"])
    assert result_show.exit_code == 0
    assert "Groq" in result_show.stdout
    assert "OpenRouter" in result_show.stdout
    assert "gsk_" in result_show.stdout

    # Test 'parallax config set groq <key>'
    result_set = runner.invoke(app, ["config", "set", "groq", "gsk_newkey9999"])
    assert result_set.exit_code == 0
    assert "Groq API Key saved" in result_set.stdout
    assert os.environ.get("GROQ_API_KEY") == "gsk_newkey9999"

    # Test 'parallax key' alias
    result_key = runner.invoke(app, ["key"])
    assert result_key.exit_code == 0
    assert "Groq" in result_key.stdout


@pytest.mark.asyncio
async def test_repl_key_command(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    agent = build_default_agent()
    console = Console(record=True, theme=PARALLAX_THEME)
    repl = InteractiveRepl(agent=agent, console=console)

    # Test /key status display
    with patch("sys.stdin.isatty", return_value=False):
        await repl._handle_command("/key")
    out = console.export_text()
    assert "provider api keys" in out

    # Test /key groq <key>
    await repl._handle_command("/key groq gsk_livekey123")
    assert os.environ.get("GROQ_API_KEY") == "gsk_livekey123"

    # Test /key openrouter <key>
    await repl._handle_command("/key openrouter sk-or-livekey456")
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-livekey456"


def test_renderer_abstracted_output_and_errors():
    console = Console(record=True, theme=PARALLAX_THEME)
    renderer = CliRenderer(console=console, verbose=False)

    # Test tool completion with large noisy output (should NOT dump lines in normal mode)
    renderer.handle_event(AgentEvent(type=EventType.RUN_STARTED, data={"task": "list files"}))
    renderer.handle_event(AgentEvent(type=EventType.TOOL_STARTED, data={"tool_call_id": "t1", "tool_name": "bash", "arguments": {"command": "ls /"}}))
    renderer.handle_event(AgentEvent(type=EventType.TOOL_COMPLETED, data={"tool_call_id": "t1", "tool_name": "bash", "result": "\n".join([f"file_{i}" for i in range(500)])}))

    output = console.export_text()
    assert "bash" in output
    assert "file_499" not in output  # Raw stdout suppressed in non-verbose mode

    # Test tool failure with Errno 22 (abstracted clean 1-line error)
    renderer.handle_event(AgentEvent(type=EventType.TOOL_STARTED, data={"tool_call_id": "t2", "tool_name": "search_code", "arguments": {"query": "test"}}))
    renderer.handle_event(AgentEvent(type=EventType.TOOL_FAILED, data={
        "tool_call_id": "t2",
        "tool_name": "search_code",
        "error": "Something went wrong, tool execution failed with error: [Errno 22] Invalid argument"
    }))

    output_failed = console.export_text()
    assert "search_code" in output_failed
    assert "invalid argument" in output_failed
    assert "Something went wrong, tool execution failed" not in output_failed




