from __future__ import annotations

import pytest

from parallax.core.context import (
    ContextConfig,
    ContextManager,
    TokenEstimator,
    UsageBaseline,
    compact,
    group_into_turns,
    load_instructions,
    truncate_output,
)
from parallax.core.context.compactor import SUMMARY_START, build_digest, merge_summary
from parallax.core.types import Message, ToolCall


def tc(id: str, name: str = "bash", **args) -> ToolCall:
    return ToolCall(id=id, name=name, arguments=args)


def assistant_calls(*calls: ToolCall) -> Message:
    return Message(role="assistant", content="", tool_calls=list(calls))


def tool_msg(id: str, content: str = "ok") -> Message:
    return Message(role="tool", content=content, tool_call_id=id)


def conversation(steps: int, parallel: int = 1) -> list[Message]:
    msgs = [
        Message(role="system", content="SYS"),
        Message(role="user", content="TASK"),
    ]
    for i in range(steps):
        calls = [tc(f"c{i}_{j}", command=f"cmd{i}_{j}") for j in range(parallel)]
        msgs.append(assistant_calls(*calls))
        msgs.extend(tool_msg(c.id, f"out{i}") for c in calls)
    return msgs


def assert_valid(msgs: list[Message]) -> None:
    """Provider invariants: results follow their call; no consecutive assistants."""
    for i, m in enumerate(msgs):
        if m.role == "tool":
            j = i
            while msgs[j].role == "tool":
                j -= 1
            owner = msgs[j]
            assert owner.role == "assistant" and owner.tool_calls
            assert m.tool_call_id in {c.id for c in owner.tool_calls}
        if m.role == "assistant" and i > 0:
            assert msgs[i - 1].role != "assistant"
        if m.role == "assistant" and m.tool_calls:
            ids = {c.id for c in m.tool_calls}
            following = set()
            k = i + 1
            while k < len(msgs) and msgs[k].role == "tool":
                following.add(msgs[k].tool_call_id)
                k += 1
            assert ids == following


# --- tokens ------------------------------------------------------------------


def test_token_estimator_basic():
    est = TokenEstimator()
    assert est.count_text("") == 0
    assert est.count_text(None) == 0
    assert est.count_text("hello world") > 0


def test_token_estimator_heuristic_fallback(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    est = TokenEstimator()
    assert est.count_text("a" * 370) == 100  # ceil(370 / 3.7)


def test_token_estimator_counts_tool_call_arguments():
    est = TokenEstimator()
    plain = Message(role="assistant", content="")
    with_call = assistant_calls(tc("1", command="x" * 400))
    assert est.count_message(with_call) > est.count_message(plain) + 20


def test_count_messages_grows_with_messages():
    est = TokenEstimator()
    a = conversation(1)
    b = conversation(5)
    assert est.count_messages(b) > est.count_messages(a)


# --- truncation --------------------------------------------------------------


def test_truncate_short_output_unchanged():
    assert truncate_output("hello", 100) == "hello"
    exact = "x" * 100
    assert truncate_output(exact, 100) == exact


def test_truncate_long_output_keeps_head_tail_and_notes_omission():
    text = "HEAD" + "m" * 50_000 + "TAIL"
    out = truncate_output(text, 1_000)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "characters omitted" in out
    assert len(out) < 1_300
    omitted = len(text) - 1_000
    assert f"[{omitted} characters omitted" in out


# --- instructions ------------------------------------------------------------


def test_instructions_none_when_missing(tmp_path):
    assert load_instructions(tmp_path) is None


def test_instructions_priority_agents_over_claude(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("claude rules")
    (tmp_path / "AGENTS.md").write_text("agents rules")
    out = load_instructions(tmp_path)
    assert out is not None and "agents rules" in out and "AGENTS.md" in out
    assert "claude rules" not in out


def test_instructions_falls_back_to_parallax_dir(tmp_path):
    (tmp_path / ".parallax").mkdir()
    (tmp_path / ".parallax" / "instructions.md").write_text("local rules")
    out = load_instructions(tmp_path)
    assert out is not None and "local rules" in out


def test_instructions_skips_empty_and_binary(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n  ")
    (tmp_path / "CLAUDE.md").write_bytes(b"\xff\xfe\x00bad")
    (tmp_path / ".parallax").mkdir()
    (tmp_path / ".parallax" / "instructions.md").write_text("good")
    out = load_instructions(tmp_path)
    assert out is not None and "good" in out


def test_instructions_are_size_capped(tmp_path):
    (tmp_path / "AGENTS.md").write_text("z" * 50_000)
    out = load_instructions(tmp_path)
    assert out is not None and len(out) < 8_200


# --- turn grouping -----------------------------------------------------------


def test_group_into_turns_keeps_parallel_results_together():
    msgs = conversation(2, parallel=3)
    turns = group_into_turns(msgs)
    assert [t[0].role for t in turns] == ["system", "user", "assistant", "assistant"]
    assert [len(t) for t in turns] == [1, 1, 4, 4]


def test_group_into_turns_final_assistant_is_own_turn():
    msgs = conversation(1) + [Message(role="assistant", content="done")]
    turns = group_into_turns(msgs)
    assert len(turns) == 4 and turns[-1][0].content == "done"


# --- compaction --------------------------------------------------------------


def test_compact_noop_when_short():
    msgs = conversation(2)  # system, user, 2 work turns; keep 2 => nothing to cut
    out, n = compact(msgs, keep_recent_turns=2)
    assert n == 0 and out is msgs


def test_compact_preserves_system_task_and_recent_turns():
    msgs = conversation(6)
    out, n = compact(msgs, keep_recent_turns=2)

    assert n == 4
    assert out[0].content == "SYS"
    assert out[1].role == "user" and out[1].content.startswith("TASK")
    assert SUMMARY_START in out[1].content
    # last two turns are verbatim
    assert out[2:] == msgs[-4:]
    assert_valid(out)


def test_compact_never_splits_parallel_tool_results():
    msgs = conversation(5, parallel=3)
    out, _ = compact(msgs, keep_recent_turns=1)
    assert_valid(out)
    assert len(out) == 2 + 4  # system, task, last assistant + 3 results


def test_compact_leaves_no_consecutive_assistant_messages():
    msgs = conversation(8) + [Message(role="assistant", content="done")]
    out, _ = compact(msgs, keep_recent_turns=2)
    assert_valid(out)


def test_compact_second_round_merges_not_stacks():
    msgs = conversation(6)
    once, _ = compact(msgs, keep_recent_turns=2)
    grown = once + [
        m for i in range(6, 10)
        for m in (assistant_calls(tc(f"c{i}", command=f"cmd{i}")), tool_msg(f"c{i}"))
    ]
    twice, n = compact(grown, keep_recent_turns=2)

    assert n > 0
    assert twice[1].content.count(SUMMARY_START) == 1
    assert twice[1].content.startswith("TASK")
    assert "cmd0_0" in twice[1].content and "cmd7" in twice[1].content
    assert_valid(twice)


def test_compact_summary_is_size_capped_with_omission_note():
    msgs = conversation(60)
    out, _ = compact(msgs, keep_recent_turns=2, summary_cap=400)
    block = out[1].content.split(SUMMARY_START)[1]
    assert len(block) < 600
    assert "earlier steps omitted" in block
    assert "cmd57_0" in block  # newest steps survive; oldest dropped


def test_omitted_counts_accumulate_across_rounds():
    msgs = conversation(40)
    once, _ = compact(msgs, keep_recent_turns=2, summary_cap=300)
    grown = once + [
        m for i in range(40, 60)
        for m in (assistant_calls(tc(f"c{i}", command=f"cmd{i}")), tool_msg(f"c{i}"))
    ]
    twice, _ = compact(grown, keep_recent_turns=2, summary_cap=300)
    assert twice[1].content.count("earlier steps omitted") == 1


def test_compact_requires_system_then_user():
    msgs = [Message(role="user", content="x")] + conversation(8)[1:]
    out, n = compact(msgs, keep_recent_turns=2)
    assert n == 0


def test_digest_reports_success_and_failure():
    turns = [
        [
            assistant_calls(tc("a", "bash", command="pytest"), tc("b", "search_code", query="x")),
            tool_msg("a", "Error: command failed badly"),
            tool_msg("b", "Found 1 match"),
        ]
    ]
    lines = build_digest(turns)
    assert any("bash(command=pytest) -> failed: Error: command failed" in ln for ln in lines)
    assert any("search_code(query=x) -> ok" in ln for ln in lines)


def test_digest_truncates_long_arguments_and_includes_text_turns():
    long_arg = "q" * 500
    turns = [
        [assistant_calls(tc("a", "bash", command=long_arg)), tool_msg("a")],
        [Message(role="assistant", content="thinking out loud")],
    ]
    lines = build_digest(turns)
    assert all(len(ln) < 200 for ln in lines)
    assert any("assistant: thinking out loud" in ln for ln in lines)


def test_merge_summary_preserves_task_text():
    task = Message(role="user", content="Fix the bug\nwith details")
    merged = merge_summary(task, ["- a -> ok"], cap=1000)
    assert merged.content.startswith("Fix the bug\nwith details")
    assert merged.role == "user"


# --- manager -----------------------------------------------------------------


def test_manager_system_prompt_with_and_without_instructions(tmp_path):
    (tmp_path / "AGENTS.md").write_text("use uv")
    with_inst = ContextManager(ContextConfig(workspace=str(tmp_path)))
    out = with_inst.build_system_prompt("BASE")
    assert out.startswith("BASE") and "use uv" in out

    off = ContextManager(ContextConfig(workspace=str(tmp_path), load_instructions=False))
    assert off.build_system_prompt("BASE") == "BASE"

    empty = ContextManager(ContextConfig(workspace=str(tmp_path / "nothing")))
    assert empty.build_system_prompt("BASE") == "BASE"


def test_manager_truncates_with_configured_limit():
    mgr = ContextManager(ContextConfig(max_tool_chars=100))
    assert len(mgr.truncate_tool_output("x" * 5000)) < 300


def test_current_tokens_uses_provider_baseline_plus_newer_estimate():
    mgr = ContextManager()
    msgs = conversation(3)
    base = UsageBaseline(tokens=1000, message_count=4)
    expected_newer = sum(mgr.estimator.count_message(m) for m in msgs[4:])
    assert mgr.current_tokens(msgs, base) == 1000 + expected_newer


def test_current_tokens_falls_back_when_baseline_stale():
    mgr = ContextManager()
    msgs = conversation(2)
    stale = UsageBaseline(tokens=99999, message_count=len(msgs) + 5)
    assert mgr.current_tokens(msgs, stale) == mgr.estimator.count_messages(msgs)
    assert mgr.current_tokens(msgs, None) == mgr.estimator.count_messages(msgs)


def test_maybe_compact_below_threshold_is_noop():
    mgr = ContextManager(ContextConfig(max_context_tokens=100_000))
    msgs = conversation(10)
    out, info = mgr.maybe_compact(msgs)
    assert out is msgs and info is None


def test_maybe_compact_triggers_above_threshold_and_reports():
    mgr = ContextManager(ContextConfig(max_context_tokens=200, compaction_threshold=0.5))
    msgs = conversation(10)
    out, info = mgr.maybe_compact(msgs)
    assert info is not None
    assert info["turns_summarized"] == 8
    assert info["compacted_tokens"] < info["original_tokens"]
    assert_valid(out)


def test_maybe_compact_trusts_provider_baseline_over_estimate():
    mgr = ContextManager(ContextConfig(max_context_tokens=10_000, compaction_threshold=0.8))
    msgs = conversation(10)  # tiny by estimate
    assert mgr.maybe_compact(msgs)[1] is None
    out, info = mgr.maybe_compact(msgs, UsageBaseline(9_000, len(msgs)))
    assert info is not None and info["original_tokens"] >= 9_000


def test_maybe_compact_noop_when_nothing_to_cut():
    mgr = ContextManager(ContextConfig(max_context_tokens=10, compaction_threshold=0.1))
    msgs = conversation(2)
    out, info = mgr.maybe_compact(msgs)
    assert out is msgs and info is None


def test_maybe_compact_flags_still_over_budget():
    mgr = ContextManager(ContextConfig(max_context_tokens=50, compaction_threshold=0.5))
    msgs = conversation(4)
    msgs[-1] = tool_msg(msgs[-1].tool_call_id or "", "w " * 2000)
    _, info = mgr.maybe_compact(msgs)
    assert info is not None and info["still_over_budget"] is True


@pytest.mark.parametrize("keep", [1, 2, 3])
def test_compact_valid_for_various_keep_values(keep):
    out, _ = compact(conversation(9, parallel=2), keep_recent_turns=keep)
    assert_valid(out)


def test_digest_keeps_preview_of_successful_results():
    # Regression: a live run lost "232 core/run.py" after compaction because
    # successful steps were summarized as just "-> ok".
    turns = [
        [
            assistant_calls(tc("a", "bash", command="wc -l core/*.py")),
            tool_msg("a", "Command succeeded:\n 12 core/a.py\n232 core/run.py\n244 total"),
        ]
    ]
    (line,) = build_digest(turns)
    assert "232 core/run.py" in line
    assert "\n" not in line
