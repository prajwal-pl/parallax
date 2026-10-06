import asyncio
from dotenv import load_dotenv

from core.agent import Agent
from core.environment import LocalEnvironment
from core.tools.bash import BashTool
from core.tools.editor import EditorTool
from core.tools.search import SearchTool

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

    result = await agent.run(prompt)

    # Log tools executed and trajectory details
    print(f"--- Execution Trajectory ({result.iterations} iterations) ---")
    for msg in result.messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"🔧 Tool Called: {tc.name}")
                print(f"   Arguments:   {tc.arguments}")
        elif msg.role == "tool":
            lines = msg.content.splitlines()
            preview = "\n   ".join(lines[:6])
            if len(lines) > 6:
                preview += f"\n   ... [{len(lines) - 6} more lines omitted]"
            print(f"📄 Tool Output ({msg.tool_call_id}):\n   {preview}\n")

    print(f"--- Final Answer ---\n{result.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
