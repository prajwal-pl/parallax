import asyncio

from core.errors import MaxIterationsError, ToolExecutionError, ToolNotFoundError
from core.llm import call_llm
from core.tool import extract_schema
from core.types import AgentResult, Message


class Agent:
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.tool_registry = {}
        self.tool_schemas = []
        self.messages = [Message(role="system", content=self.system_prompt)]

    def add_tool(self, func):
        schema = extract_schema(func)
        self.tool_registry[func.__name__] = func
        self.tool_schemas.append(schema.to_dict())

    async def _dispatch_tool(self, tool_call) -> Message:
        name = tool_call.name
        arguments = tool_call.arguments

        try:
            if name not in self.tool_registry:
                raise ToolNotFoundError(f"Tool {name} is not registered")
            func = self.tool_registry[name]
            result = await asyncio.wait_for(asyncio.to_thread(func(**arguments)), 30.0)
        except ToolExecutionError as e:
            result = f"Tool Execution failed with error: {e}"
        except Exception as e:  # noqa: BLE001
            result = (
                f"Something went wrong! Dispatch of tool {name} failed with error: {e}"
            )

        return Message(role="tool", content=str(result), tool_call_id=tool_call.id)

    MAX_ITERATIONS = 20

    async def run(self, user_input: str) -> AgentResult:
        self.messages.append(Message(role="user", content=user_input))
        iterations = 0
        while True:
            iterations += 1

            if iterations > self.MAX_ITERATIONS:
                raise MaxIterationsError(
                    f"Agent exceeded {self.MAX_ITERATIONS} without a final answer."
                )

            response = await call_llm(
                model=self.model, messages=self.messages, tools=self.tool_schemas
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
                    iterations=iterations,
                    messages=self.messages,
                )
