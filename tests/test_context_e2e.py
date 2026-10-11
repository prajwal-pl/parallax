"""End-to-end: real Agent + real tools + scripted LLM that validates every request."""

from __future__ import annotations

import pytest

from core.agent import Agent
from core.context import ContextConfig
from core.environment import LocalEnvironment
from core.tools.bash import BashTool
from core.tools.editor import EditorTool
from core.tools.search import SearchTool
from core.types import EventType, LLMResponse, Message, ToolCall, Usage


class ValidatingLLM:
    """Scripted LLM. Fails the test if any request violates provider invariants."""

    def __init__(self, script, usage_tokens: int | None = None):
        self.script = list(script)
        self.requests: list[list[Message]] = []
        self.usage_tokens = usage_tokens

    async def generate(self, model, messages, tools):
        self._validate(messages)
        self.requests.append(list(messages))
        resp = self.script.pop(0)
        if self.usage_tokens is not None and resp.usage is None:
            resp = LLMResponse(
                content=resp.content,
                tool_calls=resp.tool_calls,
                usage=Usage(self.usage_tokens, 10),
            )
        return resp

    @staticmethod
    def _validate(messages: list[Message]) -> None:
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        for i, m in enumerate(messages):
            if m.role == "assistant" and i:
                assert messages[i - 1].role != "assistant", "consecutive assistants"
            if m.role == "assistant" and m.tool_calls:
                ids = {c.id for c in m.tool_calls}
                k, seen = i + 1, set()
                while k < len(messages) and messages[k].role == "tool":
                    seen.add(messages[k].tool_call_id)
                    k += 1
                assert ids == seen, "tool call/result mismatch"
            if m.role == "tool":
                assert i > 0 and messages[i - 1].role in ("assistant", "tool")


def call(id: str, name: str, **args) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=[ToolCall(id=id, name=name, arguments=args)])


def make_agent(tmp_path, llm, **cfg) -> Agent:
    env = LocalEnvironment(workspace=str(tmp_path))
    agent = Agent(
        model="m",
        system_prompt="BASE PROMPT",
        llm=llm,
        context=ContextConfig(workspace=str(tmp_path), **cfg),
    )
    agent.add_tool(EditorTool(env=env).str_replace_editor)
    agent.add_tool(BashTool(env=env).bash)
    agent.add_tool(SearchTool(env=env).search_code)
    return agent


@pytest.mark.asyncio
async def test_repo_instructions_reach_the_model(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Always run pytest with uv.")
    llm = ValidatingLLM([LLMResponse(content="done")])
    agent = make_agent(tmp_path, llm)
    await agent.run("task")

    system = llm.requests[0][0].content
    assert system.startswith("BASE PROMPT")
    assert "Always run pytest with uv." in system


@pytest.mark.asyncio
async def test_huge_tool_output_truncated_in_history_but_full_in_event(tmp_path):
    llm = ValidatingLLM(
        [
            call("c1", "bash", command="python3 -c \"print('A'*20000 + 'THE_END')\""),
            LLMResponse(content="done"),
        ]
    )
    agent = make_agent(tmp_path, llm, max_tool_chars=2_000)
    events = [e async for e in agent.run_stream("go")]

    completed = next(e for e in events if e.type == EventType.TOOL_COMPLETED)
    assert len(completed.data["result"]) > 20_000  # UI gets everything

    tool_in_history = next(m for m in llm.requests[1] if m.role == "tool")
    assert len(tool_in_history.content) < 2_500
    assert "characters omitted" in tool_in_history.content
    assert tool_in_history.content.endswith("THE_END")  # tail (exit info) preserved


@pytest.mark.asyncio
async def test_small_outputs_untouched(tmp_path):
    llm = ValidatingLLM([call("c1", "bash", command="echo hello"), LLMResponse(content="ok")])
    agent = make_agent(tmp_path, llm)
    await agent.run("go")
    tool_msg = next(m for m in llm.requests[1] if m.role == "tool")
    assert "hello" in tool_msg.content and "omitted" not in tool_msg.content


@pytest.mark.asyncio
async def test_compaction_triggers_midrun_with_valid_history_and_events(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    script = []
    for i in range(8):
        script.append(call(f"c{i}", "bash", command=f"echo step{i}"))
    script.append(LLMResponse(content="all done"))
    llm = ValidatingLLM(script)

    agent = make_agent(
        tmp_path,
        llm,
        max_context_tokens=300,
        compaction_threshold=0.5,
        keep_recent_turns=2,
    )
    events = [e async for e in agent.run_stream("do 8 steps")]
    types = [e.type for e in events]

    assert EventType.CONTEXT_COMPACTED in types
    assert types[-1] == EventType.RUN_COMPLETED
    # compaction is announced before the model call it affects
    i = types.index(EventType.CONTEXT_COMPACTED)
    assert types[i + 1] == EventType.MODEL_STARTED

    info = events[i].data
    assert info["compacted_tokens"] < info["original_tokens"]
    assert info["turns_summarized"] > 0

    # the model actually saw a shorter history after compaction
    assert min(len(r) for r in llm.requests[3:]) < max(len(r) for r in llm.requests)
    final_request = llm.requests[-1]
    assert "Earlier progress summary" in final_request[1].content
    assert "echo step0" in final_request[1].content  # early work remembered


@pytest.mark.asyncio
async def test_provider_usage_drives_compaction(tmp_path):
    # Messages are tiny, but the provider says the prompt is huge -> must compact.
    script = [call(f"c{i}", "bash", command=f"echo {i}") for i in range(5)]
    script.append(LLMResponse(content="done"))
    llm = ValidatingLLM(script, usage_tokens=9_500)
    agent = make_agent(tmp_path, llm, max_context_tokens=10_000, compaction_threshold=0.9)
    events = [e async for e in agent.run_stream("go")]
    assert EventType.CONTEXT_COMPACTED in [e.type for e in events]


@pytest.mark.asyncio
async def test_no_compaction_when_under_budget(tmp_path):
    script = [call(f"c{i}", "bash", command=f"echo {i}") for i in range(4)]
    script.append(LLMResponse(content="done"))
    llm = ValidatingLLM(script)
    agent = make_agent(tmp_path, llm)  # default 32k budget
    events = [e async for e in agent.run_stream("go")]
    assert EventType.CONTEXT_COMPACTED not in [e.type for e in events]


@pytest.mark.asyncio
async def test_baseline_reset_after_compaction_prevents_compaction_storm(tmp_path):
    script = [call(f"c{i}", "bash", command=f"echo {i}") for i in range(12)]
    script.append(LLMResponse(content="done"))
    # provider keeps claiming a big prompt, but after compaction the stale baseline
    # must not re-trigger on every following iteration.
    llm = ValidatingLLM(script, usage_tokens=9_000)
    agent = make_agent(
        tmp_path, llm, max_context_tokens=10_000, compaction_threshold=0.85, keep_recent_turns=2
    )
    events = [e async for e in agent.run_stream("go")]
    compactions = [e for e in events if e.type == EventType.CONTEXT_COMPACTED]
    iterations = len([e for e in events if e.type == EventType.MODEL_STARTED])
    assert 1 <= len(compactions) < iterations


@pytest.mark.asyncio
async def test_failed_tool_steps_survive_in_summary(tmp_path):
    script = [call("c0", "str_replace_editor", command="view", path="missing.py")]
    script += [call(f"c{i}", "bash", command=f"echo {i}") for i in range(1, 7)]
    script.append(LLMResponse(content="done"))
    llm = ValidatingLLM(script)
    agent = make_agent(
        tmp_path, llm, max_context_tokens=300, compaction_threshold=0.4, keep_recent_turns=2
    )
    await agent.run("go")
    summary = llm.requests[-1][1].content
    assert "str_replace_editor" in summary and "failed" in summary


@pytest.mark.asyncio
async def test_result_messages_still_valid_after_compaction(tmp_path):
    script = [call(f"c{i}", "bash", command=f"echo {i}") for i in range(8)]
    script.append(LLMResponse(content="final answer"))
    llm = ValidatingLLM(script)
    agent = make_agent(tmp_path, llm, max_context_tokens=300, compaction_threshold=0.5)
    result = await agent.run("go")
    assert result.content == "final answer"
    ValidatingLLM._validate(result.messages)
    assert result.messages[-1].content == "final answer"


@pytest.mark.asyncio
async def test_multiple_runs_do_not_share_context_state(tmp_path):
    llm1 = ValidatingLLM([LLMResponse(content="one")])
    agent = make_agent(tmp_path, llm1)
    r1 = await agent.run("first")
    agent.llm = ValidatingLLM([LLMResponse(content="two")])  # type: ignore[assignment]
    r2 = await agent.run("second")
    assert r1.content == "one" and r2.content == "two"
    assert len(r2.messages) == 3  # system, user, assistant only


@pytest.mark.asyncio
async def test_search_and_editor_tools_flow_with_context_manager(tmp_path):
    (tmp_path / "mod.py").write_text("def hello():\n    return 1\n")
    llm = ValidatingLLM(
        [
            call("c1", "search_code", query="def hello"),
            call("c2", "str_replace_editor", command="str_replace", path="mod.py",
                 old_str="return 1", new_str="return 2"),
            call("c3", "bash", command="python3 -c 'import mod; print(mod.hello())'"),
            LLMResponse(content="changed to 2"),
        ]
    )
    agent = make_agent(tmp_path, llm)
    result = await agent.run("change hello to return 2")
    assert (tmp_path / "mod.py").read_text().count("return 2") == 1
    assert any(m.role == "tool" and "2" in m.content for m in result.messages)
    assert result.content == "changed to 2"
