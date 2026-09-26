import json
from core.tool import extract_schema
from core.llm import call_llm
from core.types import Message

class Agent:

    def __init__(self, model:str, system_prompt:str):
        self.model = model
        self.system_prompt = system_prompt
        self.tool_registry = {}
        self.tool_schemas = []
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    def add_tool(self, func):
        schema = extract_schema(func)
        self.tool_registry[func.__name__] = func
        self.tool_schemas.append(schema)

    def run(self, user_input: str):
        self.messages.append(
        {
            "role": "user",
            "content": user_input
        })
        while True:
            response = call_llm(model=self.model,
                messages=self.messages,
                tools=self.tool_schemas)

            if response.tool_calls:
                self.messages.append(response.model_dump())

                for tool_call in response.tool_calls:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    func = self.tool_registry[name]
                    result = func(**arguments)

                    self.messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id
                    })

                continue

            else:
                self.messages.append(response.model_dump())
                return response.content
