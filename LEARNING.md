# Parallax — Prerequisites & Learning Roadmap

> This document is a self-study guide designed specifically for building Parallax.
> It covers every concept you said you're unfamiliar with, explains them from first principles,
> and ties each one back to where it appears in the project.
> Work through these in order — each section builds on the previous one.

---

## How to Use This Document

Each section follows this structure:
1. **What it is** — plain English explanation
2. **Why it matters for Parallax** — where it shows up in the project
3. **What to build to learn it** — a concrete mini-exercise
4. **What to read** — specific resources, not generic "Google it"

Don't try to learn everything before building. The right rhythm is:
**read → understand the concept → implement the next Parallax feature that uses it → repeat.**

---

## Section 1: Python Foundations You Must Be Solid On First

Before everything else, make sure these are second nature.

### 1.1 Type hints and type checking

You're already writing type hints. Make sure you understand:

```python
# Basic
def greet(name: str) -> str: ...

# Optional values
def find(id: int) -> str | None: ...  # Python 3.10+ style

# Collections
def run(messages: list[str]) -> dict[str, int]: ...

# Callables (functions as arguments)
from collections.abc import Callable
def register(func: Callable[[str, int], bool]) -> None: ...

# TypeVar — for when input and output type must match
from typing import TypeVar
T = TypeVar("T")
def first(items: list[T]) -> T: return items[0]
```

**Install Pyright** (`pip install pyright`) and run it on your code. It will catch bugs before they reach runtime.

**Why it matters for Parallax:** Every function signature in the runtime is a contract. Types make those contracts enforceable at edit time.

---

### 1.2 Dataclasses — in depth

You've started using them. Here's what you need to fully understand:

```python
from dataclasses import dataclass, field

# Basic — Python generates __init__, __repr__, __eq__
@dataclass
class Point:
    x: float
    y: float

# Optional fields — ALWAYS use field(default=...) for optional
@dataclass
class Message:
    role: str
    content: str
    tool_calls: list | None = None       # OK — None is immutable
    metadata: dict = field(default_factory=dict)  # REQUIRED for mutable defaults

# Immutable dataclass (frozen)
@dataclass(frozen=True)
class Config:
    model: str
    max_tokens: int = 4096
```

**The crucial rule:** Never use `items: list = []` as a default in a dataclass. Every instance shares the same list object. Use `field(default_factory=list)` instead.

**Mini-exercise:** Implement all five dataclasses from your `types.py` plan — `Message`, `ToolCall`, `ToolSchema`, `AgentResult`, `AgentEvent` — completely, with proper defaults. Try constructing them in a scratch script and printing them.

---

### 1.3 Decorators — deeply

You're using `@tool`. Make sure you understand what's happening:

```python
# A decorator is just a function that takes a function and returns a function
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("before")
        result = func(*args, **kwargs)
        print("after")
        return result
    return wrapper

@my_decorator
def hello():
    print("hello")

# Exactly equivalent to:
hello = my_decorator(hello)
```

**Decorators with arguments** (like `@tool(timeout=30)`):

```python
def tool(timeout: int = 10):
    def decorator(func):
        func._is_tool = True
        func._timeout = timeout
        return func
    return decorator

@tool(timeout=30)
def slow_search(query: str) -> str: ...
```

**`functools.wraps`** — always use it when wrapping, so the original function's name and docstring survive:

```python
import functools

def log_calls(func):
    @functools.wraps(func)  # preserves func.__name__, func.__doc__
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

**Why it matters for Parallax:** The `@tool` decorator, future `@eval_task`, middleware hooks — all decorators.

---

## Section 2: asyncio — Start Here, Go Deep

This is the most important prerequisite. Get comfortable here before anything else async in the project.

### 2.1 What asyncio actually is

Python normally runs one thing at a time. When you call `requests.get("https://example.com")`, your entire program is blocked — frozen — waiting for the network.

`asyncio` is a way to write code that can **pause** while waiting for I/O (network, disk, sleep) and **let other things run** in the meantime. It's not true parallelism (that's threads/multiprocessing) — it's **cooperative concurrency**: tasks yield control voluntarily when they're waiting.

The core mental model:

```
Event loop: one thread running many coroutines
Coroutine: a function that can pause itself with `await`
Task: a coroutine scheduled on the event loop
```

```python
import asyncio

# `async def` defines a coroutine — a function that CAN be paused
async def fetch(url: str) -> str:
    # `await` means: pause here, let other things run, resume when done
    await asyncio.sleep(1)  # simulates network wait
    return f"result from {url}"

async def main():
    # Run two fetches CONCURRENTLY — not one after another
    result1, result2 = await asyncio.gather(
        fetch("https://api-a.com"),
        fetch("https://api-b.com"),
    )
    print(result1, result2)

# Entry point
asyncio.run(main())
```

Without asyncio: `2 seconds` (1s + 1s sequential).
With asyncio: `1 second` (both wait simultaneously).

### 2.2 The vocabulary you must know

| Term | What it is |
|---|---|
| `coroutine` | An `async def` function. Calling it returns a coroutine object, doesn't run it yet. |
| `await` | Pause this coroutine until the awaited thing finishes. |
| `Task` | A coroutine that's been scheduled on the event loop. Runs independently. |
| `Event loop` | The scheduler that runs coroutines and switches between them at `await` points. |
| `asyncio.run()` | The entry point. Creates an event loop, runs one coroutine, closes the loop. |
| `asyncio.gather()` | Run multiple coroutines concurrently and wait for all of them. |
| `asyncio.create_task()` | Schedule a coroutine to run but don't wait for it yet. |
| `asyncio.wait_for()` | Run a coroutine with a timeout — raises `TimeoutError` if it takes too long. |
| `asyncio.TaskGroup` | (Python 3.11+) Like `gather` but with better error handling. |

### 2.3 The cardinal rules

**1. Never block the event loop:**
```python
# BAD — freezes everything for 5 seconds
async def bad():
    import time
    time.sleep(5)  # BLOCKING — nothing else can run

# GOOD
async def good():
    await asyncio.sleep(5)  # yields control, other tasks run
```

**2. For blocking code you can't avoid (e.g., reading a file synchronously, calling a sync library):**
```python
import asyncio

async def read_file_async(path: str) -> str:
    # Run blocking code in a thread pool so it doesn't block the event loop
    return await asyncio.to_thread(open(path).read)
```

**3. `async` is contagious — once you go async, callers must also be async:**
```python
async def call_llm(): ...

async def run_agent():
    response = await call_llm()  # caller must be async too

# You can't do this:
def sync_function():
    response = await call_llm()  # SyntaxError
```

### 2.4 asyncio.TaskGroup — the modern way to run concurrent tasks

```python
async def run_tools_in_parallel(tool_calls: list) -> list:
    results = []
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(execute_tool(tc)) for tc in tool_calls]
    # All tasks are done here
    return [t.result() for t in tasks]
```

If any task raises an exception, `TaskGroup` cancels all others. This is what you want.

### 2.5 Timeouts — always use them on external calls

```python
async def call_tool_with_timeout(func, args, timeout=30.0):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, **args),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return f"Error: tool timed out after {timeout}s"
```

### 2.6 Mini-exercise: Build an async agent loop

Take your current synchronous `Agent.run()` and convert it to `async def run()`. Steps:
1. Change `def run(...)` → `async def run(...)`
2. Change `call_llm(...)` → `await call_llm(...)` (and make `call_llm` async too, using `send_async`)
3. Change `func(**arguments)` → `await asyncio.to_thread(func, **arguments)` for blocking tools
4. Add `asyncio.wait_for(...)` around tool execution
5. Run with `asyncio.run(agent.run(...))`

**Resources:**
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html) — read "Coroutines and Tasks"
- [Real Python asyncio guide](https://realpython.com/async-io-python/) — excellent walkthrough

---

## Section 3: httpx — Async HTTP

`requests` is synchronous and blocks the event loop. `httpx` is the async-first HTTP library.

```python
import httpx

# Synchronous (fine for scripts)
with httpx.Client() as client:
    response = client.get("https://api.example.com/data")
    print(response.json())

# Async (what you'll use in Parallax)
async def fetch(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()  # raises on 4xx/5xx
        return response.json()
```

Key concepts:
- `raise_for_status()` — raises `httpx.HTTPStatusError` on bad responses
- Timeouts — always set them: `httpx.Timeout(connect=5.0, read=30.0)`
- Retry — `httpx` doesn't retry by default; you write retry logic yourself

**Why it matters for Parallax:** When you build a provider-agnostic model layer (instead of using the OpenRouter SDK directly), you'll make raw HTTP calls to LLM APIs. OpenAI-compatible APIs all use the same endpoint shape.

---

## Section 4: MCP — Model Context Protocol

### 4.1 What MCP is

MCP is a **standardized protocol** for connecting AI agents to external tools and data sources. Think of it as "USB for AI tools" — instead of every tool having a custom integration, MCP defines a standard interface.

Before MCP, every agent had its own tool integration system. If you built a Slack tool for one agent framework, you couldn't reuse it in another. MCP solves this.

### 4.2 The architecture

```
MCP Client (your agent)  ←→  Transport  ←→  MCP Server (tool provider)
```

An MCP server is just a process that:
1. Exposes a list of **tools** (with schemas)
2. Exposes a list of **resources** (files, data sources)
3. Accepts tool call requests and returns results

The transport can be:
- **stdio** — the client spawns the server as a subprocess, communicates via stdin/stdout (local)
- **HTTP/SSE** — the server runs separately, client connects over the network (remote)

### 4.3 What happens at the protocol level

```
Client → Server: {"method": "tools/list"}
Server → Client: {"tools": [{"name": "read_file", "inputSchema": {...}}, ...]}

Client → Server: {"method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "main.py"}}}
Server → Client: {"result": {"content": [{"type": "text", "text": "...file contents..."}]}}
```

It's JSON-RPC over a transport. You're not building this protocol — you're using a library that handles it. But understanding the wire format helps you debug.

### 4.4 How Parallax uses MCP

Your agent discovers tools from an MCP server and adds them to its `ToolRegistry` exactly like local tools:

```python
# Local tool
agent.add_tool(read_file)

# MCP tool — the runtime discovers it from the server and registers it
await agent.add_mcp_server("filesystem-server")
# Now the agent can call read_file, write_file, etc. from the MCP server
```

**What to read:**
- [MCP specification](https://spec.modelcontextprotocol.io) — read the Overview and Core Architecture sections
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — look at the client examples

**Mini-exercise:** Run an existing MCP server locally (the filesystem server is a good start). Write a script that connects to it using the Python SDK, lists its tools, and calls one. This is how you'll understand what your agent needs to do.

---

## Section 5: Context Management in Agent Harnesses

### 5.1 The core problem

LLMs have a context window — a maximum number of tokens they can see at once. GPT-4o is 128k tokens. Claude 3.5 is 200k. These sound large but fill up quickly in a coding agent:

```
System prompt:           ~500 tokens
Task description:        ~100 tokens
File 1 contents:       ~2,000 tokens
File 2 contents:       ~3,000 tokens
Tool results (search): ~5,000 tokens
Conversation history:  ~8,000 tokens
...
Total after 10 tools: easily 30,000+ tokens
```

After many iterations, the context overflows. You can't just keep appending forever.

### 5.2 What a context manager does

It's the component that decides **what goes into the model's context on each call**. It answers: "given everything that's happened so far, what does the model actually need to see right now to make the next decision?"

```python
class ContextManager:
    def __init__(self, token_budget: int):
        self.token_budget = token_budget   # e.g. 100_000

    def build(self, state: AgentRun) -> list[Message]:
        """
        Given the full run state, return the messages list
        that fits within the token budget.
        """
        messages = []
        budget_remaining = self.token_budget

        # System prompt is always included
        messages.append(state.system_message)
        budget_remaining -= count_tokens(state.system_message)

        # Recent messages are always included
        recent = state.messages[-10:]
        for msg in reversed(recent):
            cost = count_tokens(msg)
            if budget_remaining - cost < 0:
                break
            messages.insert(1, msg)  # insert after system prompt
            budget_remaining -= cost

        # Fill remaining budget with older messages or retrieved context
        ...

        return messages
```

### 5.3 Context compaction / summarization

When the history gets too long, you summarize the older part:

```python
async def compact(self, old_messages: list[Message]) -> Message:
    """Summarize a block of old messages into one summary message."""
    summary = await self.model.generate(
        messages=[
            Message(role="system", content="Summarize this conversation history briefly."),
            *old_messages
        ]
    )
    return Message(role="assistant", content=f"[Summary of previous work]: {summary.content}")
```

The agent now sees: `[system prompt] [summary] [recent messages]` instead of the entire history.

### 5.4 Repository instructions injection

Your agent needs repo-specific context. The convention (used by Claude, Cursor, GitHub Copilot) is a special file at the repo root:

```
.parallax/instructions.md   (your convention)
CLAUDE.md                   (Claude's convention)
AGENTS.md                   (OpenAI's convention)
```

The context manager reads this file and injects it into every model call, just below the system prompt. It's like a per-project system prompt extension.

### 5.5 Token counting

You need to know how many tokens a message costs before sending it. Each model has its own tokenizer:

- OpenAI models: use `tiktoken` library
- Anthropic Claude: use `anthropic.count_tokens()`
- For approximation: `len(text) / 4` (rough estimate — 1 token ≈ 4 characters)

---

## Section 6: Evals — How to Know If Your Agent Works

### 6.1 The problem with vibe-checking

Without evals, your only feedback is "it seems to work" or "it seems broken." That's not enough. You might:
- Make a change thinking it improves the agent, but actually break a case that was working
- Not know which model is better for your use case
- Not know if your context compaction broke something

### 6.2 What an eval is

An eval is an **automated test for agent behavior**. A minimal eval looks like:

```python
@dataclass
class EvalTask:
    name: str
    task: str                    # what you ask the agent
    expected: str | None         # expected substring in output
    grader: Callable | None      # custom pass/fail function

task = EvalTask(
    name="simple_math",
    task="What is 234 * 567?",
    expected="132678",
)

result = await agent.run(task.task)
passed = task.expected in result.content
```

### 6.3 Graders — how you evaluate agent output

For coding agents, the gold standard graders are:

| Grader type | How it works | Example |
|---|---|---|
| **String match** | Does output contain expected text? | Check for a number in a math result |
| **Test execution** | Run the actual tests in the repo | `pytest` passes/fails |
| **File diff** | Did the agent make the right file changes? | Compare modified files to expected |
| **LLM judge** | Use another LLM to grade the output | "Is this a correct solution? yes/no" |

Test execution graders are the most reliable for coding agents because they use objective ground truth (the tests either pass or not).

### 6.4 What to build

A minimal eval harness:

```python
async def run_eval(agent: Agent, task: EvalTask) -> EvalResult:
    start = time.time()
    result = await agent.run(task.task)
    duration = time.time() - start

    passed = task.grader(result) if task.grader else task.expected in result.content

    return EvalResult(
        task=task.name,
        passed=passed,
        duration=duration,
        iterations=result.iterations,
        content=result.content,
    )
```

**What to read:**
- Look at how [SWE-bench](https://www.swebench.com) works — it's the industry standard benchmark for coding agents. Understanding it will show you what production evals look like.

---

## Section 7: Hybrid Retrieval and Chunking — Demystified

You said you understand basic RAG. Let's build on that.

### 7.1 What basic RAG does (your starting point)

```
Documents → Embed → Vector DB → Query → Embed query → Find nearest → Return chunks → LLM
```

Problem: vector similarity alone misses exact keyword matches. If someone searches for a specific function name like `authenticate_user`, embedding search might return semantically similar but wrong results. Keyword search finds it exactly.

### 7.2 Chunking — the part everyone gets wrong

In basic RAG, people chunk by character count or line count:
```
chunk 1: lines 1-50
chunk 2: lines 51-100
...
```

**This is wrong for code.** A function that's 60 lines long gets split in the middle, so neither chunk contains a complete function. The LLM gets broken, useless code.

**Code-aware chunking** respects semantic boundaries:
```
chunk 1: entire function `authenticate_user()` (lines 1-45)
chunk 2: entire function `logout()` (lines 47-63)
chunk 3: entire class `UserSession` (lines 65-120)
```

Tree-sitter (the library) can parse any file into an AST and find these boundaries:

```python
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def chunk_python_file(source: str) -> list[str]:
    tree = parser.parse(source.encode())
    root = tree.root_node
    chunks = []
    for node in root.children:
        if node.type in ("function_definition", "class_definition"):
            chunks.append(source[node.start_byte:node.end_byte])
    return chunks
```

### 7.3 Hybrid retrieval — vector + keyword search together

```
Query: "find the authenticate_user function"
         │
         ├── BM25 (keyword search) → finds "authenticate_user" literally
         │
         └── Vector search (semantic) → finds related auth concepts
                   │
                   ▼
            Combine scores (Reciprocal Rank Fusion)
                   │
                   ▼
            Reranking (pick top K most relevant)
                   │
                   ▼
            Return to context manager
```

**BM25** is a classic keyword ranking algorithm. In Python: `rank_bm25` library.

**Reciprocal Rank Fusion (RRF)** — a simple formula to merge ranked lists:

```python
def reciprocal_rank_fusion(rankings: list[list[str]], k=60) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

**Reranking** — after retrieval, use a cross-encoder model to more accurately score relevance:

```python
# A cross-encoder takes (query, document) pairs and scores relevance
# Much more accurate than bi-encoder similarity but too slow to run on thousands of docs
# So: retrieve 100 with hybrid search → rerank top 100 → return top 10

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, doc) for doc in candidates])
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top_10 = [doc for doc, _ in ranked[:10]]
```

### 7.4 For Parallax specifically

Start simple. In Phase 6 you don't need full hybrid retrieval immediately:

1. **First**: just `grep` / `ripgrep` for exact text search — covers 80% of use cases
2. **Second**: add Tree-sitter to chunk by function/class
3. **Third**: add vector search with `sqlite-vec` for semantic queries
4. **Fourth**: combine them with RRF for hybrid
5. **Fifth**: add a reranker if retrieval quality isn't good enough

Don't implement all of this at once. Each step improves quality. The first step is already useful.

---

## Section 8: SQLite and Local Persistence

SQLite is the right choice for local Parallax storage. No server, no setup, one file.

### 8.1 Basic SQLite in Python

```python
import sqlite3

conn = sqlite3.connect("parallax.db")
conn.row_factory = sqlite3.Row  # lets you access columns by name

conn.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        task TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
""")

conn.execute(
    "INSERT INTO runs VALUES (?, ?, ?, ?)",
    ("run-123", "Fix auth tests", "completed", "2026-09-26T20:00:00")
)
conn.commit()

row = conn.execute("SELECT * FROM runs WHERE id = ?", ("run-123",)).fetchone()
print(row["task"])  # "Fix auth tests"
```

### 8.2 Async SQLite with aiosqlite

```python
import aiosqlite

async def save_run(run_id: str, task: str):
    async with aiosqlite.connect("parallax.db") as db:
        await db.execute(
            "INSERT INTO runs (id, task, status) VALUES (?, ?, ?)",
            (run_id, task, "running")
        )
        await db.commit()

async def get_run(run_id: str) -> dict | None:
    async with aiosqlite.connect("parallax.db") as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
```

### 8.3 sqlite-vec — vector search in SQLite

`sqlite-vec` adds vector similarity search as a SQLite extension:

```python
import sqlite3
import sqlite_vec

conn = sqlite3.connect("parallax.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)  # loads the extension

# Create a vector table
conn.execute("""
    CREATE VIRTUAL TABLE chunks USING vec0(
        embedding FLOAT[1536]  -- dimension of your embeddings
    )
""")

# Insert a chunk with its embedding
embedding = get_embedding("def authenticate_user(...)...")  # list of 1536 floats
conn.execute("INSERT INTO chunks(rowid, embedding) VALUES (?, ?)", (chunk_id, embedding))

# Search for nearest neighbours
results = conn.execute("""
    SELECT rowid, distance
    FROM chunks
    WHERE embedding MATCH ?
    ORDER BY distance LIMIT 5
""", (query_embedding,)).fetchall()
```

`sqlite-vec` is lightweight and requires no separate server — perfect for local-first Parallax.

---

## Section 9: Guardrails — What the Word Actually Means

"Guardrails" is a buzzword that means **constraints on what the agent can do and say**. There are two categories:

### 9.1 Output guardrails (what the LLM says)

- Does the response contain harmful content?
- Does the response stay on topic?
- Is the response in the expected format?

For Parallax (a coding agent for your own use), output guardrails are less critical. You care more about:

### 9.2 Action guardrails (what the agent does)

This is the important one for a coding agent. Constraints on tool execution:

**Path validation** — agent can only read/write inside the workspace:
```python
def validate_path(path: str, workspace: str) -> Path:
    resolved = (Path(workspace) / path).resolve()
    workspace_path = Path(workspace).resolve()
    if not str(resolved).startswith(str(workspace_path)):
        raise PermissionError(f"Path outside workspace: {path}")
    return resolved
```

**Command allowlist** — agent can only run permitted shell commands:
```python
ALLOWED_COMMANDS = {"pytest", "python", "git", "rg", "cat", "ls"}

def validate_command(command: str) -> bool:
    binary = command.split()[0]
    if binary not in ALLOWED_COMMANDS:
        raise PermissionError(f"Command not allowed: {binary}")
```

**Iteration limit** — agent can't loop forever:
```python
if self.iterations >= self.max_iterations:
    raise MaxIterationsError(...)
```

**Confirmation prompts** — for destructive operations (git reset, file deletion), ask the user before executing:
```python
async def delete_file(path: str) -> str:
    if not await confirm(f"Delete {path}? (y/n)"):
        return "Cancelled by user"
    Path(path).unlink()
    return f"Deleted {path}"
```

Guardrails in Parallax = mostly **action validation in your tools**. You don't need a third-party library for this — it's just validation code in each tool function.

---

## Learning Order — A Concrete Sequence

Work through these in order. Each one unlocks the next Parallax feature:

```
Week 1-2: asyncio
  └── Read Real Python asyncio guide
  └── Build: convert Agent.run() to async
  └── Build: async tool execution with timeouts
  └── Milestone: agent.run() is fully async and working

Week 2-3: dataclasses + types
  └── Complete types.py (all 5 dataclasses)
  └── Build: ToolRegistry class
  └── Build: errors.py hierarchy
  └── Milestone: Agent uses typed data throughout, no raw dicts

Week 3-4: Agent vs AgentRun separation + error handling
  └── Build: AgentRun class
  └── Build: tool error handling (never crash the loop)
  └── Build: iteration guard
  └── Milestone: robust agent loop that handles failures gracefully

Week 4-5: httpx + provider abstraction
  └── Read: httpx docs (async client section)
  └── Build: LLMClient class that makes raw HTTP calls to OpenAI-compatible API
  └── Milestone: agent works without the OpenRouter SDK, using raw httpx

Week 5-6: coding tools
  └── Build: read_file, write_file, list_files, search_code
  └── Build: shell tool with timeout + path validation
  └── Build: run_tests tool
  └── Milestone: agent can read a repo, modify files, run tests

Week 6-7: events and streaming
  └── Build: AgentEvent dataclass and EventType enum
  └── Build: run_stream() that yields events
  └── Milestone: you can subscribe to events as the agent runs

Week 7-8: SQLite persistence
  └── Read: aiosqlite docs
  └── Build: save runs and messages to SQLite
  └── Build: load and inspect previous runs
  └── Milestone: runs are persisted and inspectable

Week 8-9: context management
  └── Build: token counting utility
  └── Build: basic ContextManager that trims to budget
  └── Build: history compaction with summarization
  └── Milestone: agent handles long conversations without overflowing context

Week 9-10: repository intelligence
  └── Install tree-sitter
  └── Build: code-aware chunking for Python files
  └── Build: symbol search (find function definitions)
  └── Milestone: agent navigates repos structurally, not just by reading files

Week 10-12: evals + RAG
  └── Build: minimal eval harness
  └── Write 5 eval tasks for your agent
  └── Build: sqlite-vec integration for semantic search
  └── Build: hybrid retrieval (ripgrep + vector)
  └── Milestone: you can measure agent quality and improve it systematically

Week 12+: MCP, subagents, CLI
  └── Run an MCP server locally, connect to it
  └── Build: MCP tool bridge into ToolRegistry
  └── Build: CLI with Typer + Rich
```

---

## What "AI Engineer" Actually Means (for your job search context)

The job you're aiming for requires demonstrating you can:

1. **Build agent loops from scratch** — not just call `LangChain`. You're doing this.
2. **Integrate LLM APIs** — tool calling, streaming, structured output. You're learning this.
3. **Handle production concerns** — async, error handling, retries, timeouts. This document covers it.
4. **Understand context and retrieval** — how to fit information into a context window, when to use RAG. Above.
5. **Evaluate and improve agent behavior** — evals, not vibes. Above.
6. **Write clean Python** — types, dataclasses, packaging, tests. Throughout.

**What makes Parallax a strong portfolio project:**
- It's a *runtime*, not just a script — shows architecture thinking
- Multi-interface design (CLI, web, desktop) — shows systems thinking
- You built the tool loop yourself instead of wrapping a framework — shows you understand what frameworks are abstracting
- It has evals — shows engineering discipline
- It targets a real problem (coding agent) — shows product thinking

The README you wrote for this project is already better than most AI engineer portfolio projects. The code needs to catch up to it. That's what this learning plan enables.

---

## Recommended Resources

### asyncio
- [Real Python: Async IO in Python](https://realpython.com/async-io-python/) — best practical guide
- [Python asyncio official docs](https://docs.python.org/3/library/asyncio.html) — reference

### Python best practices
- [Hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/) — project setup and tooling
- [Architecture Patterns with Python](https://www.oreilly.com/library/view/architecture-patterns-with/9781492052197/) — domain modeling, ports/adapters

### LLM / Agent engineering
- [Anthropic's guide to tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — how tool calling actually works at the API level
- [OpenAI function calling docs](https://platform.openai.com/docs/guides/function-calling) — same concept, different format
- [Simon Willison's LLM notes](https://simonwillison.net) — the best blog for practical LLM engineering

### MCP
- [MCP specification](https://spec.modelcontextprotocol.io) — the actual protocol
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — client examples

### Evals
- [SWE-bench paper](https://arxiv.org/abs/2310.06770) — read the methodology section to understand coding agent evals
- [Braintrust eval guide](https://www.braintrust.dev/docs) — practical eval framework patterns

### Retrieval
- [BM25 explained simply](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [Reciprocal Rank Fusion paper](https://dl.acm.org/doi/10.1145/1571941.1572114) — 2 pages, read it

### Tree-sitter
- [Tree-sitter Python bindings](https://github.com/tree-sitter/py-tree-sitter)
- [tree-sitter-python grammar](https://github.com/tree-sitter/tree-sitter-python)
