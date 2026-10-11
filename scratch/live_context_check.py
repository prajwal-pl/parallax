import asyncio
from dotenv import load_dotenv

from core.agent import Agent
from core.context import ContextConfig
from core.environment import LocalEnvironment
from core.tools.bash import BashTool
from core.tools.search import SearchTool
from core.types import EventType

load_dotenv()


async def main():
    env = LocalEnvironment(workspace=".")
    agent = Agent(
        model="cohere/north-mini-code:free",
        system_prompt="You are a careful engineer. Use one tool call at a time. Be brief.",
        context=ContextConfig(workspace=".", max_context_tokens=1200, compaction_threshold=0.5,
                              keep_recent_turns=2, max_tool_chars=1500),
    )
    agent.add_tool(BashTool(env=env).bash)
    agent.add_tool(SearchTool(env=env).search_code)

    prompt = ("Do these one at a time: 1) run `wc -l core/*.py`, 2) run `ls tests`, "
              "3) search_code for 'class AgentRun', 4) run `cat core/run.py`, "
              "5) run `git log --oneline -n 3`. Then tell me the first task of this list and "
              "the line count of core/run.py.")
    counts = {}
    async for e in agent.run_stream(prompt):
        counts[e.type.value] = counts.get(e.type.value, 0) + 1
        if e.type == EventType.CONTEXT_COMPACTED:
            print("COMPACTED", e.data)
        elif e.type == EventType.MODEL_COMPLETED:
            print("model call: in_tokens=", e.data["input_tokens"])
        elif e.type == EventType.TOOL_STARTED:
            print("tool:", e.data["tool_name"], e.data["arguments"])
        elif e.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
            print(e.type.value, str(e.data)[:600])
    print(counts)

asyncio.run(main())
