from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from rich.console import Console
from typer.testing import CliRunner

from parallax.cli.app import app
from parallax.cli.repl import InteractiveRepl, SlashCommandCompleter
from parallax.core.agent import Agent
from parallax.core.session import Session, SessionStore
from parallax.core.types import AgentEvent, EventType, Message, ToolCall, ToolFunction

runner = CliRunner()


def test_session_creation_and_update():
    session = Session(
        id="sess_test_123",
        workspace="/tmp/test",
        model="llama-3.3-70b-versatile",
    )
    assert session.id == "sess_test_123"
    assert session.title == "New session"
    assert session.turn_count == 0
    assert len(session.messages) == 0

    # Updating with messages
    messages = [
        Message(role="system", content="You are an assistant."),
        Message(role="user", content="Fix the authentication bug in auth.py"),
        Message(role="assistant", content="I will inspect auth.py."),
    ]
    session.update(messages, model="openai/gpt-oss-120b")

    assert session.turn_count == 1
    assert session.title == "Fix the authentication bug in auth.py"
    assert session.model == "openai/gpt-oss-120b"
    assert len(session.messages) == 3


def test_session_serialization_roundtrip():
    tool_call = ToolCall(
        id="call_1",
        name="bash",
        arguments={"command": "ls"},
    )
    session = Session(
        id="sess_roundtrip_456",
        workspace="/workspace/app",
        model="groq/llama-3.3-70b",
        title="Check directory listing",
        messages=[
            Message(role="user", content="List directory contents"),
            Message(role="assistant", content="", tool_calls=[tool_call]),
            Message(role="tool", content="file1.txt\nfile2.txt", tool_call_id="call_1"),
        ],
    )

    data = session.to_dict()
    restored = Session.from_dict(data)

    assert restored.id == session.id
    assert restored.workspace == session.workspace
    assert restored.model == session.model
    assert restored.title == session.title
    assert restored.turn_count == 1
    assert len(restored.messages) == 3
    assert restored.messages[1].tool_calls is not None
    assert restored.messages[1].tool_calls[0].name == "bash"
    assert restored.messages[1].tool_calls[0].arguments == {"command": "ls"}
    assert restored.messages[2].tool_call_id == "call_1"


def test_message_from_dict_flat_and_nested_tool_calls():
    # Standard nested format
    standard = {
        "role": "assistant",
        "content": "Running test",
        "tool_calls": [
            {
                "id": "tc_1",
                "type": "function",
                "function": {"name": "search_code", "arguments": '{"query": "def foo"}'},
            }
        ],
    }
    msg1 = Message.from_dict(standard)
    assert msg1.tool_calls is not None
    assert msg1.tool_calls[0].name == "search_code"
    assert msg1.tool_calls[0].arguments == {"query": "def foo"}

    # Flat format tolerance
    flat = {
        "role": "assistant",
        "content": "Running test",
        "tool_calls": [
            {
                "id": "tc_2",
                "type": "function",
                "name": "bash",
                "arguments": {"command": "pytest"},
            }
        ],
    }
    msg2 = Message.from_dict(flat)
    assert msg2.tool_calls is not None
    assert msg2.tool_calls[0].name == "bash"
    assert msg2.tool_calls[0].arguments == {"command": "pytest"}


def test_session_store_lifecycle(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)

    # 1. Create session
    s1 = store.create_session(workspace="/code/proj1", model="llama-3.3-70b-versatile")
    assert (tmp_path / f"{s1.id}.json").exists()

    # 2. Get session exact match
    fetched = store.get_session(s1.id)
    assert fetched is not None
    assert fetched.id == s1.id

    # 3. Update & save
    s1.update(
        [
            Message(role="user", content="Refactor module"),
            Message(role="assistant", content="Done."),
        ]
    )
    store.save_session(s1)

    fetched_updated = store.get_session(s1.id)
    assert fetched_updated is not None
    assert fetched_updated.title == "Refactor module"
    assert fetched_updated.turn_count == 1

    # 4. Create second session
    s2 = store.create_session(workspace="/code/proj2", model="openai/gpt-oss-20b")

    # 5. List sessions
    all_sessions = store.list_sessions()
    assert len(all_sessions) == 2

    # 6. Prefix lookup
    # s1 prefix match
    prefix = s1.id[:12]
    matched = store.get_session(prefix)
    # If distinct, should find s1
    if not s2.id.startswith(prefix):
        assert matched is not None
        assert matched.id == s1.id

    # 7. Delete session
    assert store.delete_session(s1.id) is True
    assert store.get_session(s1.id) is None
    assert len(store.list_sessions()) == 1
    assert store.delete_session("nonexistent") is False


def test_session_store_handles_corrupt_files(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    s1 = store.create_session(workspace="/test", model="llama-3.3")

    # Write corrupt JSON file
    corrupt_file = tmp_path / "sess_corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")

    # Should gracefully skip corrupt file
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].id == s1.id


def test_cli_sessions_list_and_delete(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
    store = SessionStore(base_dir=tmp_path)
    s1 = store.create_session(workspace="/test", model="llama-3.3-70b-versatile")
    s1.update([Message(role="user", content="Diagnose performance issue")])
    store.save_session(s1)

    with patch("parallax.cli.app.SessionStore", return_value=store):
        # List sessions
        result = runner.invoke(app, ["sessions"])
        assert result.exit_code == 0
        assert s1.id in result.stdout
        assert "Diagnose" in result.stdout

        # Delete session
        del_result = runner.invoke(app, ["sessions", "--delete", s1.id])
        assert del_result.exit_code == 0
        assert "Deleted session" in del_result.stdout

        # Verify deletion
        assert store.get_session(s1.id) is None


def test_cli_session_aliases(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
    store = SessionStore(base_dir=tmp_path)
    s = store.create_session(workspace="/test", model="llama-3.3")
    s.update([Message(role="user", content="Unit test creation")])
    store.save_session(s)

    with patch("parallax.cli.app.SessionStore", return_value=store):
        # parallax session list
        res_list = runner.invoke(app, ["session", "list"])
        assert res_list.exit_code == 0
        assert s.id in res_list.stdout

        # parallax session delete <id>
        res_del = runner.invoke(app, ["session", "delete", s.id])
        assert res_del.exit_code == 0
        assert "Deleted session" in res_del.stdout


@pytest.mark.asyncio
async def test_repl_session_persistence_and_commands(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    agent = Agent(
        model="llama-3.3-70b-versatile",
        system_prompt="Test agent",
    )

    repl = InteractiveRepl(agent=agent, session_store=store)
    assert repl.session is not None
    assert repl.session.turn_count == 0

    # Simulate /new session
    old_id = repl.session.id
    repl._new_session()
    assert repl.session.id != old_id
    assert repl.session.turn_count == 0

    # Simulate /resume session
    repl._resume_session(old_id)
    assert repl.session.id == old_id

    # Test completer with session store
    completer = SlashCommandCompleter(
        get_agent=lambda: agent,
        get_session_store=lambda: store,
    )
    from prompt_toolkit.document import Document
    doc = Document(text="/session sess_")
    completions = list(completer.get_completions(doc, MagicMock()))
    assert len(completions) > 0
    assert any(c.text == old_id for c in completions)


def test_cli_resume_command_validation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
    store = SessionStore(base_dir=tmp_path)
    s = store.create_session(workspace="/test", model="llama-3.3")

    with patch("parallax.cli.app.SessionStore", return_value=store):
        # Invalid session ID returns exit code 1
        res_fail = runner.invoke(app, ["resume", "nonexistent_sess_id"])
        assert res_fail.exit_code == 1
        assert "not found" in res_fail.stdout

        # Valid session ID invokes repl.run
        with patch.object(InteractiveRepl, "run", new_callable=AsyncMock) as mock_run:
            res_ok = runner.invoke(app, ["resume", s.id])
            assert res_ok.exit_code == 0
            mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_repl_turn_autosave_and_clear(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    agent = Agent(
        model="llama-3.3-70b-versatile",
        system_prompt="Test agent",
    )

    repl = InteractiveRepl(agent=agent, session_store=store)
    sess_id = repl.session.id

    # Mock agent stream returning RUN_COMPLETED event with messages
    async def mock_stream(*args, **kwargs):
        yield AgentEvent(
            type=EventType.RUN_COMPLETED,
            data={
                "messages": [
                    Message(role="user", content="Deploy the app"),
                    Message(role="assistant", content="Deployment successful."),
                ]
            },
        )

    with patch.object(agent, "run_stream", side_effect=mock_stream):
        await repl._run_turn("Deploy the app")

    # Verify session was updated on disk
    reloaded = store.get_session(sess_id)
    assert reloaded is not None
    assert reloaded.turn_count == 1
    assert reloaded.title == "Deploy the app"
    assert len(reloaded.messages) == 2

    # Simulate /clear command
    with patch("parallax.cli.repl.print_hero"):
        await repl._handle_command("/clear")

    assert repl.session.id != sess_id
    assert len(repl.messages) == 0


def test_session_store_ambiguous_prefix_and_exact_match(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    # Create two sessions with common prefix
    s1 = Session(id="sess_common_prefix_111", workspace="/app", model="m1")
    s2 = Session(id="sess_common_prefix_222", workspace="/app", model="m2")
    store.save_session(s1)
    store.save_session(s2)

    # Ambiguous prefix should return None
    assert store.get_session("sess_common_prefix") is None

    # Unambiguous suffix / prefix should return exact match
    res_exact = store.get_session("sess_common_prefix_111")
    assert res_exact is not None
    assert res_exact.id == s1.id

    res_suffix1 = store.get_session("111")
    assert res_suffix1 is not None
    assert res_suffix1.id == s1.id

    res_suffix2 = store.get_session("222")
    assert res_suffix2 is not None
    assert res_suffix2.id == s2.id


def test_session_store_permissions_and_atomicity(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    s = store.create_session(workspace="/code", model="llama-3.3")

    session_path = tmp_path / f"{s.id}.json"
    assert session_path.exists()
    # Confirm 0o600 permissions
    mode = session_path.stat().st_mode & 0o777
    assert mode == 0o600

    # Rapid updates maintain valid JSON without corruption
    for i in range(10):
        s.update([Message(role="user", content=f"Message {i}")])
        store.save_session(s)
        loaded = store.get_session(s.id)
        assert loaded is not None
        assert loaded.title == "Message 0"
        assert loaded.messages[0].content == f"Message {i}"


def test_session_store_empty_directory(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    assert store.list_sessions() == []
    assert store.get_session("any_id") is None


def test_session_metadata_and_long_title_truncation():
    session = Session(
        id="sess_meta",
        workspace="/git/repo",
        model="groq/llama-3.3-70b",
        metadata={"git_branch": "feature/auth", "author": "developer"},
    )
    long_prompt = "A" * 120
    session.update([Message(role="user", content=long_prompt)])

    assert session.title == ("A" * 60) + "..."
    assert session.metadata["git_branch"] == "feature/auth"
    assert session.metadata["author"] == "developer"

    # Verify serialization of metadata
    d = session.to_dict()
    restored = Session.from_dict(d)
    assert restored.metadata == session.metadata
    assert restored.title == session.title


def test_message_from_dict_malformed_json_fallback():
    # If arguments in tool call are invalid JSON string, fallback to {}
    malformed = {
        "role": "assistant",
        "content": "calling tool",
        "tool_calls": [
            {
                "id": "tc_bad",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": "NOT_A_VALID_JSON{",
                },
            }
        ],
    }
    msg = Message.from_dict(malformed)
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "bash"
    assert msg.tool_calls[0].arguments == {}


@pytest.mark.asyncio
async def test_repl_resume_already_active_and_empty_args(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    agent = Agent(model="llama-3.3", system_prompt="Test")
    repl = InteractiveRepl(agent=agent, session_store=store)

    # Calling _resume_session with currently active ID
    with patch.object(repl.console, "print") as mock_print:
        repl._resume_session(repl.session.id)
        # Should inform that session is already active
        printed = "".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "already active" in printed


@pytest.mark.asyncio
async def test_repl_resume_filters_system_messages(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    s = store.create_session(workspace="/code", model="llama-3.3")
    s.messages = [
        Message(role="system", content="System instruction from old run"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!"),
    ]
    store.save_session(s)

    agent = Agent(model="llama-3.3", system_prompt="Test")
    repl = InteractiveRepl(agent=agent, session_store=store)
    repl._resume_session(s.id)

    # System messages should NOT be in repl.messages so AgentRun reconstructs cleanly
    assert len(repl.messages) == 2
    assert all(m.role != "system" for m in repl.messages)


@pytest.mark.asyncio
async def test_repl_model_switch_persists_to_session(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    agent = Agent(model="llama-3.3-70b-versatile", system_prompt="Test")
    repl = InteractiveRepl(agent=agent, session_store=store)
    old_model = repl.agent.model
    sess_id = repl.session.id

    # Switch model via slash command handler
    await repl._handle_command("/model openai/gpt-oss-120b")
    assert repl.agent.model == "openai/gpt-oss-120b"
    assert repl.session.model == "openai/gpt-oss-120b"

    # Verify session on disk updated
    reloaded = store.get_session(sess_id)
    assert reloaded is not None
    assert reloaded.model == "openai/gpt-oss-120b"


def test_repl_show_sessions_empty_and_populated(tmp_path: Path):
    store = SessionStore(base_dir=tmp_path)
    agent = Agent(model="llama-3.3", system_prompt="Test")
    repl = InteractiveRepl(agent=agent, session_store=store)

    # Empty store (except active session which has 0 messages, let's delete it for testing empty)
    store.delete_session(repl.session.id)
    with patch.object(repl.console, "print") as mock_print:
        repl._show_sessions()
        printed = "".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "No saved sessions found" in printed

    # Populated store
    s = store.create_session(workspace="/code", model="llama-3.3")
    with patch.object(repl.console, "print") as mock_print:
        repl._show_sessions()
        assert mock_print.called


def test_cli_resume_latest_empty_and_success(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
    store = SessionStore(base_dir=tmp_path)

    with patch("parallax.cli.app.SessionStore", return_value=store):
        # Empty sessions with resume latest -> exits 1
        res_empty = runner.invoke(app, ["resume", "latest"])
        assert res_empty.exit_code == 1
        assert "No saved sessions found" in res_empty.stdout

        # Add session
        s = store.create_session(workspace="/code", model="llama-3.3")
        with patch.object(InteractiveRepl, "run", new_callable=AsyncMock) as mock_run:
            res_ok = runner.invoke(app, ["resume", "latest"])
            assert res_ok.exit_code == 0
            mock_run.assert_awaited_once()


def test_cli_run_with_resume_updates_session(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
    store = SessionStore(base_dir=tmp_path)
    s = store.create_session(workspace="/code", model="llama-3.3")
    s.update([Message(role="user", content="Initial prompt")])
    store.save_session(s)

    # Mock agent run_stream
    async def mock_stream(*args, **kwargs):
        yield AgentEvent(
            type=EventType.RUN_COMPLETED,
            data={
                "messages": [
                    Message(role="user", content="Initial prompt"),
                    Message(role="user", content="Followup task"),
                    Message(role="assistant", content="Completed followup task"),
                ]
            },
        )

    with patch("parallax.cli.app.SessionStore", return_value=store):
        with patch("parallax.core.agent.Agent.run_stream", side_effect=mock_stream):
            res = runner.invoke(app, ["run", "--resume", s.id, "Followup task"])
            assert res.exit_code == 0

            # Verify session on disk has been updated with followup
            updated = store.get_session(s.id)
            assert updated is not None
            assert updated.turn_count == 2
            assert len(updated.messages) == 3


def test_cli_session_delete_missing_target(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
    store = SessionStore(base_dir=tmp_path)

    with patch("parallax.cli.app.SessionStore", return_value=store):
        res = runner.invoke(app, ["session", "delete"])
        assert res.exit_code == 1
        assert "Session ID required" in res.stdout


