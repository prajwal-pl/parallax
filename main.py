import inspect
from collections.abc import Callable
from dotenv import load_dotenv

load_dotenv()

from openrouter import OpenRouter
import os
import json

def tool(func):
    """
    decorator to mark a function as a tool
    """
    func.is_tool = True
    return func

openrouter = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

def extract_schema(func: Callable) -> dict:
    parameters = {}
    required = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    hints = func.__annotations__

    sig = inspect.signature(func)

    for param_name, param in sig.parameters.items():
        param_hints = hints.get(param_name, str)

        parameters[param_name] = {
            "type": type_map.get(param_hints, "string"),
            "description": param_name
        }

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required
        }
    }

def call_llm(model:str, messages:list, tools:list):
    """
    Call the LLM with the given model, messages, and tools.
    """

    chat = openrouter.chat.send(model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto")

    response = chat.choices[0].message
    print(response)
    return response

class Agent:

    def __init__(self, model:str, system_prompt:str):
        self.model = model
        self.system_prompt = system_prompt
        self.tool_schemas = {}
        self.tool_registry = []
        self.messages = []

    def add_tool(self, func):
        schema = extract_schema(func)
        self.tool_schemas[func.__name__] = func
        self.tool_registry.append(schema)

    def run(self, user_input: str):
        self.messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        self.messages.append(
        {
            "role": "user",
            "content": user_input
        })
        while True:
            response = call_llm(model=self.model,
                messages=self.messages,
                tools=self.tool_registry)

            if response.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content":response.content
                })

                for tool_call in response.tool_calls:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    func = self.tool_schemas[name]
                    result = func(**arguments)

                    self.messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id
                    })

                continue

            else:
                self.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                return response.content


def main():
    print("Hello from parallax!")


if __name__ == "__main__":
    main()  
