from __future__ import annotations

import parallax
from parallax import (
    Agent,
    AgentEvent,
    AgentResult,
    BashTool,
    ContextConfig,
    ContextManager,
    DuplicateToolError,
    EditorTool,
    Environment,
    EventType,
    ExecResult,
    LLMClient,
    LLMExecutionError,
    LLMResponse,
    LocalEnvironment,
    MaxIterationsError,
    Message,
    OpenRouterClient,
    ParallaxError,
    RetryPolicy,
    SearchTool,
    ToolCall,
    ToolError,
    ToolExecutionError,
    ToolFunction,
    ToolNotFoundError,
    ToolParameters,
    ToolSchema,
    Usage,
    call_llm,
    extract_schema,
    tool,
)
from parallax.core import (
    Agent as CoreAgent,
    LocalEnvironment as CoreLocalEnvironment,
)


def test_top_level_package_exports():
    for name in parallax.__all__:
        assert hasattr(parallax, name), f"Missing export: {name}"


def test_top_level_symbols_match_core():
    assert Agent is CoreAgent
    assert LocalEnvironment is CoreLocalEnvironment


def test_agent_instantiation_from_top_level(tmp_path):
    env = LocalEnvironment(workspace=str(tmp_path))
    editor = EditorTool(env=env)
    bash = BashTool(env=env)
    search = SearchTool(env=env)

    agent = Agent(
        model="mock-model",
        system_prompt="Test System",
        context=ContextConfig(workspace=str(tmp_path)),
    )
    agent.add_tool(editor.str_replace_editor)
    agent.add_tool(bash.bash)
    agent.add_tool(search.search_code)

    assert len(agent.tool_registry) == 3
    assert "str_replace_editor" in agent.tool_registry
    assert "bash" in agent.tool_registry
    assert "search_code" in agent.tool_registry
