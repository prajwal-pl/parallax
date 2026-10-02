from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import Agent

from core.errors import MaxIterationsError, ToolExecutionError, ToolNotFoundError
from core.llm import call_llm
from core.types import AgentResult, Message


class AgentRun:
    def __init__(self, agent: Agent, user_input: str):
        self.agent = agent
        self.messages = [
            Message(role="system", content=agent.system_prompt),
            Message(role="user", content=user_input),
        ]
        self.iterations = 0

    async def _dispatch_tool(self, tool_call) -> Message:
        name = tool_call.name
        arguments = tool_call.arguments

        try:
            if name not in self.agent.tool_registry:
                raise ToolNotFoundError(f"Tool '{name}' not registered")
            func = self.agent.tool_registry[name]
            result = await asyncio.wait_for(
                asyncio.to_thread(func, **arguments), timeout=30
            )
        except ToolExecutionError as e:
            result = f"Tool execution failed with error: {e}"
        except Exception as e:
            result = f"Something went wrong, tool execution failed with error: {e}"

        return Message(role="tool", content=str(result), tool_call_id=tool_call.id)

    MAX_ITERATIONS = 30

    async def execute(self) -> AgentResult:
        while True:
            self.iterations += 1
            if self.iterations > self.MAX_ITERATIONS:
                raise MaxIterationsError(
                    f"Agent exceeded {self.MAX_ITERATIONS} iterations and did not produce a result"
                )

            response = await call_llm(
                model=self.agent.model,
                messages=self.messages,
                tools=self.agent.tool_schemas,
            )

            if response.tool_calls:
                self.messages.append(response.to_message())
                async with asyncio.TaskGroup() as tg:
                    tasks = [
                        tg.create_task(self._dispatch_tool(tc))
                        for tc in response.tool_calls
                    ]

                for task in tasks:
                    self.messages.append(task.result())

                continue
            else:
                self.messages.append(response.to_message())
                return AgentResult(
                    content=response.content or "",
                    iterations=self.iterations,
                    messages=self.messages,
                )
