# Parallax — Comprehensive Development Plan

> **What is Parallax?**
> A modular, model-agnostic AI agent runtime. The runtime is the product.
> CLI, desktop, and web are consumers of that runtime — not the focus.
> The LLM is one component. Parallax is the system around it.

---

## Project North Star

The end goal is a runtime capable of receiving a task like:

```
Fix the failing authentication tests in this repository.
```

And autonomously: understanding the task → inspecting the repo → reading files → forming a plan → modifying files → running tests → inspecting failures → iterating → producing a final result.

The same runtime runs locally via CLI, in a desktop app, and behind a web API. No agent logic lives in any interface layer.

---

## Current Status

Phase 0 (Foundation) and Phase 1 (Minimal Agent Runtime) are partially complete.

**What works today:**
- `@tool` decorator marks functions as tools
- `extract_schema()` converts type hints + docstrings → JSON schema
- `Agent` class: registers tools, runs the tool-call loop, dispatches, returns final answer
- `call_llm()` in its own `llm.py` module
- `core/__init__.py` with clean relative imports
- System prompt initialized once in `__init__`
- `response.model_dump()` for safe message serialization
- `Message` dataclass started in `types.py`

**What still needs to happen before moving forward:**
- `Message` dataclass needs `field(default=None)` for optional fields
- `ToolCall` and `ToolSchema` dataclasses not yet created
- No `errors.py` custom exceptions
- No `ToolRegistry` class (still raw dicts on `Agent`)
- Runtime is synchronous — must become `async` before CLI/API can be built
- No iteration guard on the `while True` loop
- No tool error handling (a crashing tool crashes the whole agent)
- No structured `AgentRun` / `AgentResult` separation from `Agent`

---

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│                  Interfaces                      │
│         CLI          Desktop          Web        │
└────────────┬─────────────┬─────────────┬─────────┘
             │             │             │
             └─────────────┼─────────────┘
                           │ Runtime API (events, AgentResult)
┌──────────────────────────▼───────────────────────┐
│                  PARALLAX CORE                   │
│                                                  │
│  runtime/    → Agent, AgentRun, execution loop   │
│  models/     → LLM abstraction, provider clients │
│  tools/      → @tool, ToolRegistry, built-ins    │
│  context/    → Memory, budgets, compaction       │
│  repository/ → File tools, search, AST, Git      │
│  mcp/        → MCP client, tool/resource bridge  │
│  agents/     → Subagents, delegation             │
│  persistence/→ Sessions, runs, SQLite            │
│  evals/      → Benchmarks, graders               │
└──────────────────────────────────────────────────┘
```

**Non-negotiable rule:** `core/` has zero platform dependencies. No CLI libraries, no web frameworks, no terminal formatting. It exposes APIs and emits events. Interfaces decide how to render them.

---

## Production Python Best Practices (for this project)

Since this is your first production Python project, these apply throughout every phase:

### Code Quality

**Always annotate types.** Python's type hints are not enforced at runtime but they make your code self-documenting, catch bugs at edit time, and enable refactoring confidence.

```python
# Bad
def add_tool(self, func):
    ...

# Good
def add_tool(self, func: Callable) -> None:
    ...
```

**Use dataclasses for data, classes for behaviour.** If a class mostly holds fields and you're writing `__init__` by hand, use `@dataclass` instead. If a class has real methods and logic, it's a proper class.

**Use `field(default=None)` for optional fields in dataclasses** — never put mutable defaults (lists, dicts) directly as default values in a dataclass.

```python
# Dangerous — all instances share the same list
@dataclass
class Foo:
    items: list = []

# Correct
from dataclasses import dataclass, field

@dataclass
class Foo:
    items: list = field(default_factory=list)
```

**Custom exceptions from a base class.** Never raise bare `Exception`. Create a hierarchy:

```python
class ParallaxError(Exception): ...
class ToolNotFoundError(ParallaxError): ...
class LLMError(ParallaxError): ...
class MaxIterationsError(ParallaxError): ...
```

Catch specific exceptions, never bare `except:`.

**Use `src/` layout** when you're ready to package `parallax-core` for distribution. It prevents import-path ambiguity and is the current industry standard.

### Async

Go async early — the README is explicit about this: streaming, parallel tool calls, MCP, subagents, and cancellation all require it. A synchronous runtime cannot be patched into an async one easily.

```python
# The target shape
class Agent:
    async def run(self, task: str) -> AgentResult: ...
    async def run_stream(self, task: str) -> AsyncIterator[AgentEvent]: ...
```

Rules for async:
- Never use `time.sleep()` — use `await asyncio.sleep()`
- Never use synchronous HTTP — use `httpx.AsyncClient` or async SDK methods
- If a tool is blocking (e.g., file I/O), run it in a thread: `await asyncio.to_thread(blocking_fn, args)`
- Use `asyncio.wait_for(coro, timeout=N)` to enforce timeouts on tool calls
- Use `asyncio.TaskGroup` (Python 3.11+) when running tools in parallel

### Testing

Write tests from the start — not after. Use `pytest`. For async tests use `pytest-asyncio`.

```
tests/
├── unit/
│   ├── test_tool.py       # extract_schema, @tool decorator
│   ├── test_agent.py      # loop logic, dispatch
│   └── test_types.py      # dataclass construction
└── integration/
    └── test_agent_run.py  # full run against a mock LLM
```

Mock the LLM for unit tests — never call real APIs in tests.

### Logging

Never use `print()` in the runtime. Use Python's `logging` module:

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Tool call: %s %s", name, arguments)
logger.error("Tool failed: %s", exc, exc_info=True)
```

Each module gets its own logger via `__name__`. The interface layer configures log levels and handlers. The core just emits.

---

## Phase-by-Phase Roadmap

### Phase 0 — Foundation ✅ (mostly done)

Project structure, `uv`, `pyproject.toml`, `dataclasses`, relative imports, clean module separation. **Done.** Minor remaining item: move to `src/` layout when packaging.

---

### Phase 1 — Minimal Agent Runtime ✅ (mostly done)

Core loop working. **Remaining items in this phase:**

#### 1.1 Complete `types.py`

```python
from dataclasses import dataclass, field

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class Message:
    role: str                               # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict

@dataclass
class AgentResult:
    content: str
    iterations: int
    messages: list[Message]
```

#### 1.2 `errors.py`

```python
class ParallaxError(Exception): ...
class ToolNotFoundError(ParallaxError): ...
class LLMError(ParallaxError): ...
class MaxIterationsError(ParallaxError): ...
class ToolExecutionError(ParallaxError): ...
```

#### 1.3 `ToolRegistry` class in `tool.py`

Encapsulate the raw dict/list currently living on `Agent`:

```python
class ToolRegistry:
    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []

    def register(self, func: Callable) -> None:
        if not getattr(func, "is_tool", False):
            raise ValueError(f"{func.__name__} is not decorated with @tool")
        schema = extract_schema(func)
        self._functions[func.__name__] = func
        self._schemas.append(schema)

    def get(self, name: str) -> Callable:
        if name not in self._functions:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")
        return self._functions[name]

    def schemas(self) -> list[dict]:
        return self._schemas
```

#### 1.4 Add iteration guard + tool error handling to `Agent.run()`

```python
MAX_ITERATIONS = 20

def run(self, user_input: str) -> AgentResult:
    self.memory.add_user(user_input)
    iterations = 0

    while True:
        if iterations >= MAX_ITERATIONS:
            raise MaxIterationsError(f"Agent exceeded {MAX_ITERATIONS} iterations")

        response = call_llm(...)
        iterations += 1

        if response.tool_calls:
            self.memory.add_assistant(response)
            for tc in response.tool_calls:
                try:
                    func = self.tools.get(tc.name)
                    result = func(**tc.arguments)
                except ToolNotFoundError as e:
                    result = f"Error: Tool not found — {e}"
                except Exception as e:
                    result = f"Error: Tool '{tc.name}' failed — {e}"
                    logger.error("Tool execution failed", exc_info=True)
                self.memory.add_tool_result(tc.id, str(result))
            continue
        else:
            self.memory.add_assistant(response)
            return AgentResult(content=response.content, iterations=iterations, messages=self.memory.get_messages())
```

> [!IMPORTANT]
> Tool errors must **never crash the agent**. Return the error as a string result. The LLM sees it and can reason about it, retry differently, or report the failure. Crashing the loop means the user sees a traceback instead of a graceful response.

---

### Phase 2 — Runtime Robustness

#### 2.1 Go async

This is the most important architectural change. Everything downstream depends on it.

```python
class Agent:
    async def run(self, task: str) -> AgentResult:
        ...

    async def run_stream(self, task: str) -> AsyncIterator[AgentEvent]:
        ...
```

The LLM call becomes:
```python
response = await self.model.generate(messages=..., tools=...)
```

Tool execution becomes:
```python
# For blocking tools (file I/O, subprocess):
result = await asyncio.to_thread(func, **arguments)

# With timeout:
result = await asyncio.wait_for(asyncio.to_thread(func, **arguments), timeout=30.0)
```

#### 2.2 Agent vs AgentRun separation

Currently `Agent` holds both its configuration (model, tools, system prompt) and its execution state (messages). This is the bug: calling `run()` twice accumulates history from the first run.

```python
class Agent:
    """Reusable configuration. Create once, run many times."""
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = ToolRegistry()

    async def run(self, task: str) -> AgentResult:
        run = AgentRun(agent=self, task=task)
        return await run.execute()

class AgentRun:
    """One execution. Holds all ephemeral state for a single run."""
    def __init__(self, agent: Agent, task: str):
        self.agent = agent
        self.task = task
        self.messages: list[Message] = [
            Message(role="system", content=agent.system_prompt)
        ]
        self.iterations = 0
        self.tool_calls_made: list[ToolCall] = []

    async def execute(self) -> AgentResult:
        ...
```

#### 2.3 Cancellation support

```python
async def run(self, task: str, cancel_token: asyncio.Event | None = None) -> AgentResult:
    ...
    while True:
        if cancel_token and cancel_token.is_set():
            raise asyncio.CancelledError("Run cancelled by caller")
        ...
```

#### 2.4 Retry on LLM transient errors

```python
async def _call_with_retry(self, messages, tools, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await self.model.generate(messages=messages, tools=tools)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            await asyncio.sleep(wait)
```

---

### Phase 3 — Coding Agent Tools

The first real use-case. Build the tool set that lets the agent operate on a repository:

| Tool | What it does | Implementation |
|---|---|---|
| `read_file(path)` | Read a file's contents | `pathlib.Path.read_text()` |
| `write_file(path, content)` | Write/create a file | `pathlib.Path.write_text()` |
| `edit_file(path, old, new)` | Replace text in a file | `str.replace()` + write |
| `list_files(path, pattern)` | List files in a directory | `glob` / `pathlib` |
| `search_code(query, path)` | Text/regex search across files | `grep` via subprocess or `ripgrep` |
| `shell(command)` | Run a shell command | `asyncio.create_subprocess_exec()` |
| `git(args)` | Run a git command | `asyncio.create_subprocess_exec()` |
| `run_tests(command)` | Run the test suite | `asyncio.create_subprocess_exec()` |

> [!CAUTION]
> `shell` and `git` are dangerous. Without sandboxing, the agent can delete files, run arbitrary commands, and leak environment variables. Build these tools first behind an explicit allowlist or confirmation prompt, even before the full sandbox exists.

Shell tool pattern with timeout:

```python
@tool
async def shell(command: str) -> str:
    """Run a shell command and return stdout + stderr."""
    proc = await asyncio.create_subprocess_exec(
        *command.split(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        proc.kill()
        return "Error: command timed out after 30 seconds"

    return (stdout + stderr).decode()
```

---

### Phase 4 — Events and Streaming

The runtime becomes event-driven. Every significant action emits an event. Interfaces subscribe to the event stream.

```python
from enum import Enum
from dataclasses import dataclass

class EventType(Enum):
    RUN_STARTED = "run_started"
    MODEL_STARTED = "model_started"
    MODEL_DELTA = "model_delta"         # streaming token
    MODEL_COMPLETED = "model_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    RUN_COMPLETED = "run_completed"
    RUN_CANCELLED = "run_cancelled"
    RUN_FAILED = "run_failed"

@dataclass
class AgentEvent:
    type: EventType
    data: dict
    timestamp: float
```

Streaming API:
```python
async for event in agent.run_stream("Fix the auth tests"):
    if event.type == EventType.MODEL_DELTA:
        print(event.data["delta"], end="", flush=True)
    elif event.type == EventType.TOOL_STARTED:
        print(f"\n[calling {event.data['name']}...]")
    elif event.type == EventType.RUN_COMPLETED:
        print("\nDone.")
```

The CLI renders these in the terminal. The web API streams them over SSE. The desktop app renders them in a UI. **The runtime doesn't know or care which.**

---

### Phase 5 — Context Engineering

When the agent reads many files and makes many tool calls, the context window fills up. This phase introduces a `ContextManager` that decides what goes into each model request.

Key concepts:

**Token budgeting** — track how many tokens are in `self.messages` and enforce a ceiling. When approaching the limit, trigger compaction.

**History compaction** — summarize older messages instead of dropping them:
```python
async def compact(self, messages: list[Message], budget: int) -> list[Message]:
    # Keep system prompt + recent N messages
    # Summarize everything in between
    summary = await self.model.generate(
        messages=[...older messages...],
        system="Summarize this conversation history concisely."
    )
    return [system_msg, summary_msg, *recent_messages]
```

**Repository instructions** — a `CLAUDE.md` / `AGENTS.md` / `.parallax/instructions.md` file at the repo root gets injected into context automatically to give the agent project-specific context.

---

### Phase 6 — Repository Intelligence

Beyond raw file I/O, the agent needs to *understand* the repository:

| Layer | Technology | What it provides |
|---|---|---|
| Filesystem | `pathlib`, `glob` | File discovery, metadata |
| Text search | `ripgrep` via subprocess | Fast grep across all files |
| Structural | Tree-sitter | Parse Python/JS/TS/Go/Rust into AST |
| Symbol search | Tree-sitter queries | Find function/class definitions |
| Repository graph | Custom | File → imports → symbols → tests |

Tree-sitter enables queries like:
```python
find_definition("authenticate_user")  # → auth/service.py:42
find_references("UserModel")          # → [models.py, views.py, tests.py]
find_tests_for("auth/service.py")     # → tests/test_auth.py
```

This means the agent spends fewer tokens reading entire files when it can navigate directly to what matters.

---

### Phase 7 — RAG (Repository Search)

For large codebases, keyword + AST search is not enough. Add semantic retrieval:

```
Repository files
     ↓
Parse + Chunk (by function/class boundaries, not arbitrary lines)
     ↓
Generate embeddings (OpenAI text-embedding-3-small or local)
     ↓
Vector index (SQLite + sqlite-vec, or pgvector in platform mode)
     ↓
Hybrid retrieval (BM25 lexical + vector semantic)
     ↓
Reranking (cross-encoder or LLM-based)
     ↓
Context manager
     ↓
Model
```

Use **chunking at semantic boundaries** (function level, class level) not arbitrary line counts — this is critical for code. A chunk that splits a function in half is useless.

---

### Phase 8 — MCP Integration

Model Context Protocol gives you a standardized way to connect external tools and resources without writing bespoke integrations.

```python
# The agent treats MCP tools identically to local tools
agent.add_mcp_server("mcp://localhost:8000")  # discovers tools automatically
```

Use an established Python MCP client library — don't reimplement the protocol. The runtime's job is to bridge MCP tool schemas and results into its own `ToolRegistry` format.

---

### Phase 9 — Subagents

For complex tasks, delegate to specialized agents:

```
Main Agent
    │
    ├── Research Agent  (investigate the codebase)
    ├── Coding Agent    (implement the changes)
    └── Testing Agent   (run and verify tests)
```

Key design: subagents run in isolated `AgentRun` instances with their own message history. The orchestrator agent sees only the result of each subagent, not their full internal loop. Use `asyncio.TaskGroup` to run subagents in parallel where possible.

---

### Phase 10 — Sandboxing

Before the agent can be used on real repositories:

- **Filesystem scope** — restrict tools to the repo directory. Reject paths that escape via `../`.
- **Command allowlist** — only allow explicitly permitted shell commands.
- **Network policy** — decide whether tools can make outbound HTTP calls.
- **Resource limits** — CPU time, memory, disk write limits via `resource` module or containers.
- **Prompt injection defense** — treat file contents as untrusted data, not instructions.

```python
def validate_path(path: str, workspace: str) -> Path:
    resolved = (Path(workspace) / path).resolve()
    if not str(resolved).startswith(workspace):
        raise SecurityError(f"Path escape attempt: {path}")
    return resolved
```

---

### Phase 11 — Persistence

Sessions and runs become first-class stored objects.

Local persistence: **SQLite** via `aiosqlite`.

Schema:
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    model TEXT,
    system_prompt TEXT,
    workspace TEXT
);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    task TEXT,
    status TEXT,    -- "running" | "completed" | "failed" | "cancelled"
    result TEXT,
    iterations INTEGER,
    created_at TEXT,
    completed_at TEXT
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(id),
    role TEXT,
    content TEXT,
    created_at TEXT
);
```

The runtime writes a checkpoint after each tool call so a failed run can be inspected — or eventually resumed.

---

### Phase 12 — Evals

Every new capability needs a measurable baseline. Build a lightweight eval harness:

```python
@eval_task(
    repo="fixtures/example-project",
    task="Fix the failing authentication tests",
    grader=lambda result: "tests pass" in result.content
)
async def test_auth_fix(agent): ...
```

This lets you compare:
- Model A vs Model B on the same task
- Runtime v1 vs v2 after a refactor
- Tool strategy A vs tool strategy B

Without evals, you don't know if a change made the agent better or worse.

---

### Phase 13 — Observability

Structured telemetry across every run:

```python
@dataclass
class RunMetrics:
    run_id: str
    duration_ms: float
    iterations: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    estimated_cost_usd: float
    tool_calls: list[ToolCallMetric]
    model_calls: list[ModelCallMetric]
```

Each tool call records: name, arguments, duration, success/failure, result size.
Each model call records: context size, latency, token usage.

Use Python's `logging` module at every key event. Consider OpenTelemetry for standardized tracing once you have multiple services.

---

### Phase 14 — CLI (First Interface)

Only after the core is stable does it make sense to build the CLI.

**Technology:** Typer (for commands) + Rich (for terminal rendering).

CLI is thin:
1. Parse input
2. Configure the runtime
3. Subscribe to `AgentEvent` stream
4. Render events in the terminal

```bash
parallax "Fix the authentication tests"
parallax run --model openai/gpt-4o "Refactor the database layer"
parallax session list
parallax models
```

The CLI renders streaming tokens in real time, shows tool calls as they happen, and displays a structured summary on completion. All rendering logic lives in the CLI — not in `core/`.

---

### Phase 15 — Desktop + Web + Platform (Future)

These build on top of the stable core and event system:

- **Desktop** — Tauri (Rust shell) + React UI consuming the core as a local process
- **Web** — Next.js frontend + FastAPI backend streaming events over SSE/WebSocket
- **Platform** — Auth, org management, model gateway, usage quotas, billing

Platform concerns never enter `core/`. The core receives an effective configuration and executes. The platform decides what configuration is permitted.

---

## Recommended Immediate Next Steps

You are currently between Phase 1 and Phase 2. In order:

```
1. Complete types.py
   ├── Add field(default=None) to Message
   ├── Add ToolCall dataclass
   ├── Add ToolSchema dataclass
   └── Add AgentResult dataclass

2. Add errors.py
   └── ParallaxError, ToolNotFoundError, LLMError, MaxIterationsError, ToolExecutionError

3. Add ToolRegistry class to tool.py
   └── encapsulates _functions dict and _schemas list

4. Add iteration guard to Agent.run()
   └── raise MaxIterationsError after N iterations

5. Add tool error handling
   └── wrap dispatch in try/except, return error string to LLM

6. Go async
   └── Agent.run() → async def run()
   └── call_llm() → async (use send_async)
   └── tools that do I/O → asyncio.to_thread()

7. Separate Agent from AgentRun
   └── Agent = configuration
   └── AgentRun = one execution, holds messages

8. Write first tests
   └── test extract_schema()
   └── test ToolRegistry
   └── test Agent loop with a mock LLM

9. Build coding tools
   └── read_file, write_file, list_files, search_code, shell, run_tests
```

---

## Technology Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python | AI ecosystem, asyncio, rapid iteration |
| Dependency management | uv | Fast, modern, pyproject.toml native |
| Type checking | Pyright / Pylance | Catches bugs at edit time |
| Testing | pytest + pytest-asyncio | Industry standard, async support |
| CLI | Typer + Rich | Clean commands, beautiful terminal output |
| Async HTTP | httpx (async) | Modern, async-first, OpenAI-compatible |
| Local persistence | SQLite + aiosqlite | No server needed, async, portable |
| Platform persistence | PostgreSQL + asyncpg | Production scale, when needed |
| Vector search | sqlite-vec (local) / pgvector (platform) | Start simple, scale later |
| Code parsing | Tree-sitter | Language-aware, fast, multi-language |
| MCP | Established Python MCP client | Don't reimplement the protocol |
| Desktop | Tauri + React | Native, performant, TypeScript UI |
| Web | Next.js + FastAPI | React frontend, Python backend |
| Observability | Python logging → OpenTelemetry | Start with stdlib, standardize later |

---

## What Not to Build

Per the project's own philosophy — use mature libraries instead of reinventing:

| Don't build | Use instead |
|---|---|
| HTTP client | `httpx` |
| Git operations | `subprocess` calling `git`, or `gitpython` |
| TLS / crypto | stdlib or established libraries |
| MCP protocol | Official Python MCP SDK |
| OAuth / OIDC | `authlib` |
| Tree-sitter bindings | `tree-sitter` Python package |
| Tokenizer | Provider tokenizer or `tiktoken` |
| Vector search | `sqlite-vec` / `pgvector` |

Build the agent runtime, orchestration, context engine, and eval harness. Everything else, use what exists.

---

> [!NOTE]
> The README states this clearly: *"The goal is not to hide the mechanics behind an agent framework from day one. The core runtime will initially be implemented directly so that the underlying concepts are understood before abstractions are introduced."*
>
> This plan respects that — implement the loop, state, tool dispatch, and context management by hand. Only introduce frameworks (if ever) after you understand what they're abstracting.
