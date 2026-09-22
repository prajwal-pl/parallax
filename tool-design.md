# Parallax — Tool System Design Reference

## 1. The Starting Point: A Plain Function

The user writes a normal Python function. Nothing special:

```python
def search_web(query: str) -> str:
    """Search the web for a given query and return results."""
    return f"Results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))
```

The goal: turn these into something the LLM can **discover, choose, and call** at runtime.

---

## 2. The `@tool` Decorator — Marking a Function

The decorator is the user-facing API. All it does is **tag** the function. It doesn't wrap it, doesn't change its behavior:

```python
def tool(func):
    """Mark a function as an agent tool."""
    func._is_tool = True
    return func
```

Usage:

```python
@tool
def search_web(query: str) -> str:
    """Search the web for a given query and return results."""
    return f"Results for: {query}"
```

After decoration, `search_web` is still the exact same function — it just has `search_web._is_tool = True` stamped on it.

> [!NOTE]
> The decorator is purely for ergonomics. You could skip it entirely and just pass raw functions to `agent.add_tool()`. The decorator just makes intent explicit.

---

## 3. Schema Extraction — The Crucial Bridge (Step by Step)

The LLM never sees your Python code. It sees a **JSON schema** describing the tool. Schema extraction is the process of converting a Python function's signature into that JSON.

### 3.1 What the LLM needs to see

The LLM API (OpenAI, Gemini, etc.) expects tool definitions in this shape:

```json
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Search the web for a given query and return results.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "query"
        }
      },
      "required": ["query"]
    }
  }
}
```

This tells the LLM:
- There's a tool called `search_web`
- It does "Search the web for a given query and return results."
- It takes one parameter: `query`, which is a string, and it's required

### 3.2 Where does each piece come from?

| JSON field | Python source |
|---|---|
| `function.name` | `func.__name__` |
| `function.description` | `func.__doc__` (the docstring) |
| `parameters.properties` | `inspect.signature(func).parameters` + `func.__annotations__` |
| `parameters.required` | Parameters that have no default value |

### 3.3 The extraction function, line by line

```python
import inspect

def extract_schema(func) -> dict:
    # Maps Python types to JSON Schema type strings
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    # Step A: Get type hints
    # For `def search_web(query: str) -> str`, this gives:
    #   {'query': <class 'str'>, 'return': <class 'str'>}
    hints = func.__annotations__

    # Step B: Inspect the function signature to get parameter details
    # This gives us parameter names, default values, etc.
    sig = inspect.signature(func)

    # Step C: Build the properties dict and required list
    parameters = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Look up the type hint for this parameter, default to str
        param_type = hints.get(param_name, str)

        # Convert Python type -> JSON Schema type
        parameters[param_name] = {
            "type": type_map.get(param_type, "string"),
            "description": param_name,
        }

        # If the parameter has no default value, it's required
        # e.g. `query: str` -> required
        #      `query: str = "default"` -> optional
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    # Step D: Assemble the final schema
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

### 3.4 Worked example: tracing through `calculate`

Given this function:

```python
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))
```

Here's what happens inside `extract_schema(calculate)`:

**Step A — `func.__annotations__`**
```python
{'expression': <class 'str'>, 'return': <class 'str'>}
```

**Step B — `inspect.signature(func)`**
```
(expression: str) -> str
```
`sig.parameters` is an OrderedDict:
```python
{'expression': <Parameter "expression: str">}
```

**Step C — Loop through parameters**

Iteration 1: `param_name = "expression"`, `param = <Parameter "expression: str">`
- `hints.get("expression", str)` → `<class 'str'>`
- `type_map.get(str, "string")` → `"string"`
- `param.default` → `inspect.Parameter.empty` (no default was set)
- So `"expression"` goes into `required`

After the loop:
```python
parameters = {
    "expression": {
        "type": "string",
        "description": "expression"
    }
}
required = ["expression"]
```

**Step D — Assemble**
```json
{
  "type": "function",
  "function": {
    "name": "calculate",
    "description": "Evaluate a math expression and return the result.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "expression"
        }
      },
      "required": ["expression"]
    }
  }
}
```

### 3.5 A more complex example: optional parameters

```python
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for a given query."""
    ...
```

Tracing through:
- `query`: type `str`, no default → **required**, goes into `required`
- `max_results`: type `int`, default `5` → **optional**, does NOT go into `required`

Result:
```json
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Search the web for a given query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "query" },
        "max_results": { "type": "integer", "description": "max_results" }
      },
      "required": ["query"]
    }
  }
}
```

The LLM sees that `query` is required but `max_results` is optional — it can choose to include it or not.

---

## 4. The Agent Class — Registry + Loop

Now everything connects. The Agent holds two data structures:

- **`tool_registry`** — a dict: `name → callable` (for dispatching)
- **`tool_schemas`** — a list of JSON objects (for sending to the LLM)

```python
import json

class Agent:
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.tool_registry = {}   # {"search_web": <function>, "calculate": <function>}
        self.tool_schemas = []     # [schema_1, schema_2, ...]
        self.messages = []

    def add_tool(self, func):
        """Register a @tool-decorated function."""
        schema = extract_schema(func)           # Python func → JSON schema
        self.tool_registry[func.__name__] = func # Store for dispatch
        self.tool_schemas.append(schema)          # Store for LLM

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        while True:
            # ── 1. CALL THE LLM ──────────────────────────────────
            # Send the conversation history AND the tool schemas.
            # The LLM reads the schemas to know what it can call.
            response = call_llm(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.messages,
                ],
                tools=self.tool_schemas,
            )

            # ── 2. DID THE LLM CALL A TOOL? ─────────────────────
            if response.tool_calls:
                # The LLM chose to call one or more tools.
                # Its response looks like:
                #   tool_calls: [
                #     {id: "abc", function: {name: "calculate", arguments: '{"expression":"234*567"}'}}
                #   ]

                # Append the assistant's message to history (it contains the tool_call metadata)
                self.messages.append(response.message)

                for tool_call in response.tool_calls:
                    # Extract what the LLM wants
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    # ── 3. DISPATCH ──────────────────────────────
                    # Look up the actual Python function and call it
                    func = self.tool_registry[name]
                    result = func(**args)

                    # ── 4. FEED RESULT BACK ──────────────────────
                    # Add the tool's output to the conversation
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })

                # Loop back to step 1.
                # The LLM now sees the tool results in its context
                # and can either call more tools or give a final answer.
                continue

            else:
                # ── 5. FINAL ANSWER ──────────────────────────────
                # The LLM responded with plain text, no tool calls.
                # We're done.
                self.messages.append(response.message)
                return response.message.content
```

---

## 5. User-Facing Usage

```python
# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for a given query and return results."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))

# Create agent and register tools
agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant.")
agent.add_tool(search_web)
agent.add_tool(calculate)

# Run
answer = agent.run("What is 234 * 567?")
print(answer)
```

---

## 6. Runtime Flow — What Actually Happens

When the user asks `"What is 234 * 567?"`:

```
┌─────────────────────────────────────────────────────────────────┐
│  User: "What is 234 * 567?"                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent calls LLM with:                                         │
│    messages: [user msg]                                        │
│    tools: [search_web schema, calculate schema]                │
│                                                                 │
│  LLM reads the schemas and decides:                            │
│    "calculate looks right for this"                            │
│                                                                 │
│  LLM returns:                                                  │
│    tool_calls: [{                                              │
│      name: "calculate",                                        │
│      arguments: {"expression": "234 * 567"}                    │
│    }]                                                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent dispatches:                                             │
│    func = self.tool_registry["calculate"]                      │
│    result = func(expression="234 * 567")                       │
│    result = "132678"                                           │
│                                                                 │
│  Appends to messages:                                          │
│    {"role": "tool", "tool_call_id": "...", "content": "132678"}│
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent calls LLM again with:                                   │
│    messages: [user msg, assistant tool_call, tool result]      │
│    tools: [search_web schema, calculate schema]                │
│                                                                 │
│  LLM sees "132678" as the tool result.                         │
│  No more tools needed.                                         │
│                                                                 │
│  LLM returns: "234 × 567 = 132,678"                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent returns the final answer to the user.                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Multi-Step Tool Use

The loop handles chains naturally. If the user asks "Search for the population of France and multiply it by 2":

1. **Loop iteration 1**: LLM calls `search_web(query="population of France")` → result: `"67.75 million"`
2. **Loop iteration 2**: LLM sees the search result, calls `calculate(expression="67750000 * 2")` → result: `"135500000"`
3. **Loop iteration 3**: LLM sees the calculation result, returns `"The population of France multiplied by 2 is approximately 135.5 million."`

Each iteration, the LLM has the full conversation history — all previous tool calls and their results — so it can chain them logically.

---

## 8. Summary Table

| Piece | What it is | One-liner |
|---|---|---|
| `@tool` | Decorator | Tags a function as available to the agent |
| `extract_schema()` | Schema builder | Converts type hints + docstring → JSON the LLM reads |
| `tool_registry` | `dict[str, callable]` | Maps tool name → Python function for dispatch |
| `tool_schemas` | `list[dict]` | JSON schemas sent to the LLM API |
| `add_tool()` | Registration method | Extracts schema + stores function and schema |
| Agent loop (`while True`) | Core loop | Call LLM → dispatch tool if requested → loop or return |

> [!IMPORTANT]
> The LLM never executes anything. It only **says** "I want to call `calculate` with `{"expression": "234 * 567"}`". Your Python code does the actual execution and feeds the result back. The LLM is the decision-maker; your code is the executor.
