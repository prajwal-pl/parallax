from __future__ import annotations

import json
from itertools import chain

from parallax.core.types import Message

SUMMARY_START = "[Earlier progress summary]"
_OMIT_PREFIX = "... ["
_FAILURE_PREFIXES = ("Error:", "Tool execution failed", "Something went wrong")


def group_into_turns(messages: list[Message]) -> list[list[Message]]:
    """
    Split a conversation into indivisible turns:
    - a system message or user message is its own turn
    - an assistant message starts a turn and owns ALL tool results that follow it

    Compaction only ever cuts between turns, so a tool call is never separated
    from its results (providers reject that).
    """
    turns: list[list[Message]] = []
    for msg in messages:
        if msg.role == "tool" and turns and turns[-1][0].role == "assistant":
            turns[-1].append(msg)
        else:
            turns.append([msg])
    return turns


def _short(value: object, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe_call(name: str, arguments: dict) -> str:
    args = ", ".join(f"{k}={_short(v)}" for k, v in list(arguments.items())[:3])
    return f"{name}({args})"


def build_digest(turns: list[list[Message]]) -> list[str]:
    """Deterministic one-line-per-step description of what happened (no LLM)."""
    lines: list[str] = []
    for turn in turns:
        head = turn[0]
        if head.role != "assistant":
            lines.append(f"- {head.role}: {_short(head.content, 100)}")
            continue
        if not head.tool_calls:
            lines.append(f"- assistant: {_short(head.content, 100)}")
            continue
        results = {m.tool_call_id: m.content for m in turn[1:]}
        for tc in head.tool_calls:
            content = results.get(tc.id, "")
            failed = content.startswith(_FAILURE_PREFIXES)
            status = "failed" if failed else "ok"
            # Keep a short preview of successes too: later steps often need a
            # value from an earlier result (a count, a path, a line number).
            preview = _short(" | ".join(content.splitlines()), 160 if not failed else 80)
            detail = f": {preview}" if preview else ""
            lines.append(f"- {_describe_call(tc.name, tc.arguments)} -> {status}{detail}")
    return lines


def _split_summary(content: str) -> tuple[str, list[str]]:
    if SUMMARY_START not in content:
        return content, []
    task, _, block = content.partition(SUMMARY_START)
    return task.rstrip(), [ln for ln in block.strip().splitlines() if ln.strip()]


def merge_summary(task_message: Message, new_lines: list[str], cap: int) -> Message:
    """
    Fold digest lines into the first user message. Any previous summary block is
    replaced (old lines kept, new lines appended), never stacked. When over `cap`
    characters the oldest lines are dropped.
    """
    task, old_lines = _split_summary(task_message.content)

    prior_omitted = 0
    steps: list[str] = []
    for line in old_lines:
        if line.startswith(_OMIT_PREFIX):
            prior_omitted += int(line[len(_OMIT_PREFIX) :].split(" ")[0])
        else:
            steps.append(line)
    steps.extend(new_lines)

    omitted = prior_omitted
    while len("\n".join(steps)) > cap and len(steps) > 1:
        steps.pop(0)
        omitted += 1
    lines = ([f"{_OMIT_PREFIX}{omitted} earlier steps omitted]"] if omitted else []) + steps

    content = f"{task}\n\n{SUMMARY_START}\n" + "\n".join(lines)
    return Message(role=task_message.role, content=content)


def compact(
    messages: list[Message], keep_recent_turns: int = 2, summary_cap: int = 3_000
) -> tuple[list[Message], int]:
    """
    Summarize the middle of the conversation. Returns (messages, turns_summarized);
    turns_summarized == 0 means nothing was changed.
    Keeps: system prompt, the user task (with a merged summary), last N turns.
    """
    turns = group_into_turns(messages)
    if (
        len(turns) < 2
        or turns[0][0].role != "system"
        or turns[1][0].role != "user"
        or len(turns) <= 2 + keep_recent_turns
    ):
        return messages, 0

    middle = turns[2:-keep_recent_turns] if keep_recent_turns else turns[2:]
    recent = turns[-keep_recent_turns:] if keep_recent_turns else []
    task = merge_summary(turns[1][0], build_digest(middle), summary_cap)
    compacted = [*turns[0], task, *chain.from_iterable(recent)]
    return compacted, len(middle)
