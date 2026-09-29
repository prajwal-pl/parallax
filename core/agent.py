from core.tool import extract_schema
from core.llm import call_llm
from core.types import AgentResult, Message
from core.errors import MaxIterationsError, ToolExecutionError, ToolNotFoundError
import asyncio

class Agent:

    def __init__(self, model:str, system_prompt:str):
        self.model = model
        self.system_prompt = system_prompt
        self.tool_registry = {}
        self.tool_schemas = []
        self.messages = [
            Message(
                role = "system",
                content = self.system_prompt
            )
        ]

    def add_tool(self, func):
        schema = extract_schema(func)
        self.tool_registry[func.__name__] = func
        self.tool_schemas.append(schema.to_dict())

    MAX_ITERATIONS = 20

    async def run(self, user_input: str) -> AgentResult:
        self.messages.append(
       Message(
                role = "user",
                content = user_input
            ))
        iterations = 0
        while True:
            iterations += 1

            if iterations > self.MAX_ITERATIONS:
                raise MaxIterationsError(f"Agent exceeded {self.MAX_ITERATIONS} without a final answer.")

            response = await call_llm(model=self.model,
                messages=self.messages,
                tools=self.tool_schemas)

            if response.tool_calls:
                self.messages.append(response.to_message())

                for tool_call in response.tool_calls:
                    name = tool_call.name
                    arguments = tool_call.arguments

                    try:
                        if name not in self.tool_registry:
                            raise ToolNotFoundError(f"Tool {name} is not registered")
                        func = self.tool_registry[name]
                        result = await asyncio.to_thread(func, **arguments)
                    except ToolExecutionError as e:
                        result = f"Tool Execution failed with error: {e}"
                    except Exception as e:
                        result = f"Something went wrong! Dispatch of tool {name} failed with error: {e}"

                    self.messages.append(Message(
                        role = "tool",
                        content = str(result),
                        tool_call_id =  tool_call.id
                    ))

                continue

            else:
                self.messages.append(response.to_message())
                return AgentResult(
                    content=response.content or "",
                    iterations=iterations,
                    messages=self.messages
                )
