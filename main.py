import inspect

def tool(func):
    """
    decorator to mark a function as a tool
    """
    func.is_tool = True
    return func

def extract_schema(func: function) -> dict:
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
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required
        }
    }

class Agent:

    def run(self):
        pass

class Tools:
    def __init__(self):
        pass

    def tool_search(self, query: str):
        pass

def main():
    print("Hello from parallax!")


if __name__ == "__main__":
    main()  
