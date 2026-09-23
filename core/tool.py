from dotenv import load_dotenv

load_dotenv()

import inspect
from openrouter import OpenRouter
from collections.abc import Callable
import os

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
        "function": {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required
            }
        }
    }