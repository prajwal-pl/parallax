import asyncio

from dotenv import load_dotenv

from parallax import (
    Agent,
    BashTool,
    ContextConfig,
    EditorTool,
    EventType,
    LocalEnvironment,
    SearchTool,
)

load_dotenv()


async def main():
    env = LocalEnvironment(workspace=".")
    editor = EditorTool(env=env)
    bash_tool = BashTool(env=env)
    search_tool = SearchTool(env=env)

    agent = Agent(
        model="cohere/north-mini-code:free",
        context=ContextConfig(workspace="."),
        system_prompt=(
            "You are an expert autonomous software engineer. "
            "You have access to:\n"
            "- str_replace_editor: view, create, and modify files\n"
            "- bash: run tests, linters, and shell commands\n"
            "- search_code: search text and regex patterns across the codebase\n"
        ),
    )

    agent.add_tool(editor.str_replace_editor)
    agent.add_tool(bash_tool.bash)
    agent.add_tool(search_tool.search_code)

    prompt = "Understand the entire architecture, search for all tool definitions and explan them in brief"
    print(f"\nUser: {prompt}\n")

    print("--- Real-Time Event Stream ---")
    async for event in agent.run_stream(prompt):
        match event.type:
            case EventType.RUN_STARTED:
                print(f"🚀 Run Started: {event.data.get('task')}")

            case EventType.MODEL_STARTED:
                print(
                    f"\n🧠 Model Thinking [Iteration {event.data.get('iterations')}] "
                    f"(model: {event.data.get('model')})..."
                )

            case EventType.MODEL_COMPLETED:
                tokens = ""
                if event.data.get("input_tokens") is not None:
                    tokens = (
                        f", {event.data.get('input_tokens')} in / "
                        f"{event.data.get('output_tokens')} out tokens"
                    )
                print(
                    f"   Model Response Received "
                    f"({event.data.get('latency_ms', 0):.0f} ms{tokens})."
                )

            case EventType.MODEL_RETRY:
                print(
                    f"⏳ Retry {event.data.get('attempt')} in "
                    f"{event.data.get('delay', 0):.1f}s: {event.data.get('error')}"
                )

            case EventType.CONTEXT_COMPACTED:
                print(
                    f"🧹 Context compacted: {event.data.get('original_tokens')} → "
                    f"{event.data.get('compacted_tokens')} tokens "
                    f"({event.data.get('turns_summarized')} turns summarized)"
                )

            case EventType.TOOL_STARTED:
                print(f"🔧 Tool Started: {event.data.get('tool_name')}")
                print(f"   Arguments:   {event.data.get('arguments')}")

            case EventType.TOOL_COMPLETED:
                res = str(event.data.get("result", ""))
                lines = res.splitlines()
                preview = "\n   ".join(lines[:6])
                if len(lines) > 6:
                    preview += f"\n   ... [{len(lines) - 6} more lines omitted]"
                print(
                    f"✅ Tool Completed: {event.data.get('tool_name')} "
                    f"({event.data.get('tool_call_id')}):\n   {preview}"
                )

            case EventType.TOOL_FAILED:
                print(
                    f"❌ Tool Failed: {event.data.get('tool_name')} "
                    f"({event.data.get('tool_call_id')}):\n   {event.data.get('error')}"
                )

            case EventType.RUN_COMPLETED:
                print(
                    f"\n--- Final Answer ({event.data.get('iterations')} iterations) ---"
                )
                print(f"{event.data.get('content')}\n")

            case EventType.RUN_FAILED:
                print(f"\n💥 Run Failed: {event.data.get('error')}\n")


if __name__ == "__main__":
    asyncio.run(main())
