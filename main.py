import asyncio
from dotenv import load_dotenv

from core.agent import Agent
from core.environment import LocalEnvironment
from core.tools.bash import BashTool
from core.tools.editor import EditorTool
from core.tools.search import SearchTool
from core.types import EventType

load_dotenv()


async def main():
    env = LocalEnvironment(workspace=".")
    editor = EditorTool(env=env)
    bash_tool = BashTool(env=env)
    search_tool = SearchTool(env=env)

    agent = Agent(
        model="cohere/north-mini-code:free",
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

    prompt = "Use search_code to find where BashTool is defined and run pytest on tests/test_bash_tool.py using bash."
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
                print("   Model Response Received.")

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
