# Parallax — Architecture Review & Next Steps

## Current State

### What exists today

```
parallax/
├── core/
│   ├── agent.py    # Agent class + call_llm helper
│   └── tool.py     # @tool decorator, extract_schema, OpenRouter client
├── main.py          # Demo: web_search + calculate tools → Agent.run()
├── pyproject.toml   # Dependencies: openrouter, python-dotenv, tavily-python
└── README.md        # Design notes
```

### What works

- **`@tool` decorator** — marks functions, no wrapping
- **`extract_schema()`** — converts type hints + docstring → JSON schema for the LLM
- **`Agent` class** — registers tools, runs the tool-call loop, dispatches, returns final answer
- **`call_llm()`** — thin wrapper around `openrouter.chat.send()`
- **Demo tools** — `calculate` (eval) and `web_search` (Tavily)

### Known issues to fix first

| Issue | Where | What's wrong |
|---|---|---|
| Swapped variable names | `agent.py:23-24` | `tool_schemas` holds **functions**, `tool_registry` holds **schemas** — names are backwards |
| OpenRouter client lives in `tool.py` | `tool.py:17` | The LLM client is tangled with tool utilities — should be separate |
| `call_llm` is a loose function | `agent.py:4` | Should live on the Agent or in its own module, not floating at module level |
| System prompt re-appended on every `run()` | `agent.py:32-35` | Calling `run()` twice duplicates the system message in `self.messages` |
| No `__init__.py` in `core/` | `core/` | Package isn't properly initialized |

---

## Vision

Parallax is an **agent runtime** — a minimal core that handles the agent loop, tool management, and LLM communication. Multiple frontends (CLI, API, web, desktop) consume this core:

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   CLI    │  │   API    │  │   Web    │  │ Desktop  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     └─────────────┴──────┬──────┴─────────────┘
                          │
                    ┌─────┴─────┐
                    │   core/   │
                    │ (runtime) │
                    └───────────┘
```

---

## Proposed Core Architecture

### Target file structure

```
core/
├── __init__.py          # Public API: exports Agent, tool, etc.
├── agent.py             # Agent class — the loop, message management
├── tool.py              # @tool decorator, extract_schema, ToolRegistry class
├── llm.py               # LLM client abstraction (provider-agnostic)
├── memory.py            # Conversation history / message management
├── errors.py            # Custom exceptions (ToolNotFound, LLMError, etc.)
└── types.py             # Shared dataclasses / TypedDicts (Message, ToolCall, etc.)
```

### Why each module

| Module | Responsibility | Why it's separate |
|---|---|---|
| `agent.py` | The agent loop + orchestration | Core logic, shouldn't know about HTTP or providers |
| `tool.py` | `@tool` decorator, schema extraction, `ToolRegistry` | Self-contained — no LLM dependency |
| `llm.py` | LLM API calls, provider abstraction | Swap OpenRouter for Anthropic/OpenAI/local without touching agent.py |
| `memory.py` | Message list management, system prompt handling | Prevents bugs like double-appending system prompt |
| `errors.py` | Custom exceptions | Clean error handling across the stack |
| `types.py` | Data structures (`Message`, `ToolCall`, `ToolSchema`) | Shared vocabulary between modules |

---

## Next Steps — In Order

### Phase 1: Fix foundations (do this now)

These are cleanup tasks on the existing code. No new features, just getting the base right.

#### 1.1 Swap the variable names in Agent

```python
# Currently (wrong names):
self.tool_schemas = {}      # actually holds functions
self.tool_registry = []     # actually holds schemas

# Fix to:
self.tool_registry = {}     # name → callable (for dispatch)
self.tool_schemas = []      # list of JSON schemas (for the LLM)
```

Update `add_tool`, `run`, and dispatch logic accordingly.

#### 1.2 Add `core/__init__.py`

```python
from core.agent import Agent
from core.tool import tool
```

Clean public API — users just do `from core import Agent, tool`.

#### 1.3 Extract the LLM client into `core/llm.py`

Move OpenRouter setup out of `tool.py`. Create a simple abstraction:

```python
# core/llm.py
class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str):
        ...

    def chat(self, messages: list, tools: list) -> dict:
        """Send messages + tool schemas, return the response."""
        ...
```

The Agent takes an `LLMClient` instance instead of a raw model string. This makes it provider-agnostic.

#### 1.4 Fix the system prompt duplication

Move system prompt setup to `__init__` or guard it so `run()` can be called multiple times:

```python
def run(self, user_input: str):
    if not self.messages or self.messages[0]["role"] != "system":
        self.messages.insert(0, {"role": "system", "content": self.system_prompt})
    self.messages.append({"role": "user", "content": user_input})
    ...
```

---

### Phase 2: Build out the runtime (next)

#### 2.1 `core/types.py` — Define your data structures

Use dataclasses or TypedDicts for the core types. This gives you type safety and makes the codebase self-documenting:

```python
from dataclasses import dataclass

@dataclass
class Message:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list | None = None

@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
```

#### 2.2 `core/tool.py` — ToolRegistry class

Upgrade from raw dicts/lists to a proper registry:

```python
class ToolRegistry:
    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []

    def register(self, func: Callable):
        schema = extract_schema(func)
        self._functions[func.__name__] = func
        self._schemas.append(schema)

    def get(self, name: str) -> Callable:
        if name not in self._functions:
            raise ToolNotFoundError(f"Tool '{name}' not registered")
        return self._functions[name]

    def schemas(self) -> list[dict]:
        return self._schemas
```

#### 2.3 `core/memory.py` — Conversation management

Encapsulate message history so the Agent doesn't manage raw lists:

```python
class Memory:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user(self, content: str): ...
    def add_assistant(self, message): ...
    def add_tool_result(self, tool_call_id: str, result: str): ...
    def get_messages(self) -> list: ...
    def clear(self): ...
```

#### 2.4 `core/errors.py` — Custom exceptions

```python
class ParallaxError(Exception): ...
class ToolNotFoundError(ParallaxError): ...
class LLMError(ParallaxError): ...
class SchemaExtractionError(ParallaxError): ...
```

#### 2.5 Refactored Agent class

After all the above, the Agent becomes clean and focused:

```python
class Agent:
    def __init__(self, llm: LLMClient, system_prompt: str):
        self.llm = llm
        self.tools = ToolRegistry()
        self.memory = Memory(system_prompt)

    def add_tool(self, func):
        self.tools.register(func)

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)

        while True:
            response = self.llm.chat(
                messages=self.memory.get_messages(),
                tools=self.tools.schemas(),
            )

            if response.tool_calls:
                self.memory.add_assistant(response)
                for tc in response.tool_calls:
                    func = self.tools.get(tc.name)
                    result = func(**tc.arguments)
                    self.memory.add_tool_result(tc.id, str(result))
                continue
            else:
                self.memory.add_assistant(response)
                return response.content
```

---

### Phase 3: Advanced runtime features (later)

Once the core is solid, these are the features that make it a real runtime:

| Feature | What it does | Why it matters |
|---|---|---|
| **Streaming** | Yield tokens as they arrive instead of waiting for full response | Essential for CLI/web UX |
| **Async support** | `async def run()` with `await` | Required for API/web platforms |
| **Middleware / Hooks** | `on_tool_call`, `on_response`, `on_error` callbacks | Lets platforms add logging, UI updates, auth |
| **Max iterations guard** | Cap the while loop at N iterations | Prevents runaway tool loops |
| **Token/cost tracking** | Count tokens per run, estimate cost | Useful for API billing, user dashboards |
| **Multi-provider support** | OpenRouter, OpenAI, Anthropic, Ollama (local) | Users pick their provider |
| **Retry + error handling** | Exponential backoff on rate limits, graceful tool errors | Production resilience |
| **Conversation persistence** | Save/load conversation history to disk or DB | Needed for web/desktop sessions |

---

### Phase 4: Platform frontends (future)

Each frontend is a thin layer that imports `core` and adds platform-specific I/O:

```
cli/          → argparse/click, stdin/stdout, streaming print
api/          → FastAPI/Flask, REST endpoints, SSE streaming
web/          → Frontend (React/Svelte) + API backend
desktop/      → Electron/Tauri wrapping the web frontend
```

These all share the same `core/` runtime. None of them contain agent logic.

---

## Recommended Order of Work

```
1. Fix foundations (Phase 1)          ← do this now, ~1 hour
   ├── 1.1 Swap variable names
   ├── 1.2 Add __init__.py
   ├── 1.3 Extract LLM client
   └── 1.4 Fix system prompt bug

2. Build out the runtime (Phase 2)   ← next session, ~2-3 hours
   ├── 2.1 types.py
   ├── 2.2 ToolRegistry class
   ├── 2.3 Memory class
   ├── 2.4 errors.py
   └── 2.5 Refactor Agent

3. Advanced features (Phase 3)       ← ongoing
   ├── Streaming
   ├── Async
   ├── Hooks/middleware
   └── Multi-provider

4. Platform frontends (Phase 4)      ← after core is stable
   ├── CLI first (simplest)
   ├── API second
   ├── Web third
   └── Desktop last
```

> [!IMPORTANT]
> Keep `core/` with **zero platform dependencies**. No CLI libraries, no web frameworks, no UI. The core should be importable and usable from any context. All platform-specific code goes in its own directory.
