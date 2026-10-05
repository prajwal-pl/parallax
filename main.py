import asyncio

from dotenv import load_dotenv

from core.agent import Agent
from core.environment import LocalEnvironment
from core.tools.editor import EditorTool

load_dotenv()


async def main():
    env = LocalEnvironment(workspace=".")
    editor = EditorTool(env=env)

    agent = Agent(
        model="cohere/north-mini-code:free",
        system_prompt=(
            "You are an expert coding assistant. "
            "Use the str_replace_editor tool whenever you need to view, create, or edit files in the workspace."
        ),
    )

    agent.add_tool(editor.str_replace_editor)

    prompt = "Explain the entire architecture of tools directory as well as the core agent architecture."
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
