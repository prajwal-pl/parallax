# Parallax

> A modular, model-agnostic AI coding-agent runtime designed to power
> CLI, desktop, and web applications.

Parallax is being built as a serious AI engineering project rather than
as a single-purpose coding CLI.

The central idea is to separate the **agent runtime** from the interface
that users interact with. The runtime should understand how to reason
over a task, interact with a repository, use tools, manage context, call
different models, coordinate subagents, interact with MCP servers,
execute code safely, and maintain run state. The CLI, desktop
application, and web application should all consume the same core
runtime.

## Project Vision

Parallax aims to become a reusable **AI agent runtime** rather than
merely another coding-agent interface.

The runtime should eventually be capable of taking a task such as:

``` text
Fix the failing authentication tests in this repository.
```

and autonomously:

1.  Understand the task.
2.  Inspect the repository.
3.  Search for relevant code.
4.  Read files and configuration.
5.  Form a plan.
6.  Modify files.
7.  Run tests or other verification commands.
8.  Inspect failures.
9.  Iterate.
10. Produce a final result with a useful summary of what changed.

The same runtime should be usable through multiple interfaces:

``` text
                    ┌─────────────────────────┐
                    │      Parallax Core      │
                    │                         │
                    │     Agent Runtime       │
                    │     Model Layer         │
                    │     Tool System         │
                    │     Context Engine      │
                    │     Repository Intel    │
                    │     MCP                 │
                    │     Subagents           │
                    │     Sandbox             │
                    │     Persistence         │
                    │     Evals               │
                    │     Observability       │
                    └────────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
           CLI               Desktop              Web
        Python first        Tauri + TS/        Next.js + TS
                           Rust later
```

The core is the product's technical foundation. The interfaces are
consumers of that foundation.

------------------------------------------------------------------------

## Why Parallax Exists

AI coding agents are no longer simply chat interfaces with code
generation.

A capable coding agent needs to operate inside an environment:

-   repositories
-   files
-   terminals
-   Git
-   tests
-   build systems
-   package managers
-   documentation
-   external tools
-   model providers
-   MCP servers
-   multiple agents
-   persistent sessions
-   context limits
-   permissions and security boundaries

This makes the interesting engineering problem much larger than:

``` text
prompt → LLM → answer
```

Parallax is intended to explore the complete system:

``` text
User Task
   ↓
Agent Runtime
   ↓
Context + Repository State
   ↓
Model
   ↓
Tool Calls
   ↓
Execution Environment
   ↓
Tool Results
   ↓
State Update
   ↓
Context Rebuild
   ↓
Model
   ↓
...
   ↓
Verified Result
```

The project is also intentionally designed as a learning vehicle for
modern AI engineering.

The goal is not to hide the mechanics behind an agent framework from day
one. The core runtime will initially be implemented directly so that the
underlying concepts are understood before abstractions are introduced.

------------------------------------------------------------------------

# Core Thesis

The central architectural principle is:

> **Parallax Core should not know or care whether it is being used by a
> CLI, desktop application, web application, or another program.**

The core should provide APIs such as:

``` python
from parallax import Agent

agent = Agent(
    model="...",
    tools="auto",
)

result = await agent.run(
    "Fix the failing authentication tests."
)
```

The interface consuming the runtime decides how that result is
displayed.

For example:

``` text
CLI
 └── renders AgentEvents in the terminal

Desktop
 └── renders AgentEvents in a graphical UI

Web
 └── streams AgentEvents over WebSocket/SSE
```

This prevents UI concerns from leaking into the runtime.

------------------------------------------------------------------------

# Architecture

``` text
┌─────────────────────────────────────────────────────────────┐
│                        Interfaces                           │
│                                                             │
│        CLI             Desktop             Web              │
└───────────────┬───────────────┬───────────────┬─────────────┘
                │               │               │
                └───────────────┼───────────────┘
                                │
                         Runtime API
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                        PARALLAX CORE                         │
│                                                             │
│  Agent Runtime                                              │
│  ├── State                                                  │
│  ├── Execution Loop                                         │
│  ├── Termination                                            │
│  ├── Budgets                                                │
│  ├── Retries                                                │
│  └── Cancellation                                           │
│                                                             │
│  Model Layer                                                │
│  ├── Provider abstraction                                   │
│  ├── Streaming                                              │
│  ├── Tool calling                                           │
│  └── Usage                                                  │
│                                                             │
│  Tool System                                                │
│  ├── File tools                                             │
│  ├── Search tools                                           │
│  ├── Shell tools                                            │
│  ├── Git tools                                              │
│  └── Test tools                                             │
│                                                             │
│  Context Engine                                             │
│  ├── Context budgeting                                      │
│  ├── History                                                │
│  ├── Compaction                                             │
│  └── Repository instructions                                │
│                                                             │
│  Repository Intelligence                                    │
│  ├── File discovery                                         │
│  ├── Symbol search                                          │
│  ├── AST analysis                                           │
│  └── Code indexing                                          │
│                                                             │
│  MCP                                                        │
│  ├── Servers                                                 │
│  ├── Tools                                                   │
│  └── Resources                                               │
│                                                             │
│  Agents                                                      │
│  ├── Subagents                                               │
│  ├── Delegation                                              │
│  └── Parallel execution                                     │
│                                                             │
│  Sandbox                                                     │
│  ├── Process isolation                                       │
│  ├── Filesystem boundaries                                   │
│  └── Network permissions                                     │
│                                                             │
│  Persistence                                                 │
│  ├── Sessions                                                │
│  ├── Runs                                                    │
│  ├── Messages                                                │
│  └── Checkpoints                                             │
│                                                             │
│  Evals + Observability                                       │
└───────────────────────────────┬─────────────────────────────┘
                                │
                         Optional Platform
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                       PARALLAX PLATFORM                      │
│                                                             │
│  API                                                         │
│  Authentication / OAuth                                     │
│  Users / Organizations                                      │
│  Model access policies                                      │
│  Usage / quotas                                             │
│  Billing                                                     │
│  Metrics                                                     │
│  Cloud persistence                                           │
└─────────────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

# Repository Structure

The intended long-term structure is:

``` text
parallax/
│
├── packages/
│   ├── parallax-core/
│   │   └── src/
│   │       └── parallax/
│   │           ├── runtime/
│   │           ├── models/
│   │           ├── tools/
│   │           ├── context/
│   │           ├── repository/
│   │           ├── mcp/
│   │           ├── agents/
│   │           ├── persistence/
│   │           └── evals/
│   │
│   └── parallax-cli/
│       └── src/
│           └── parallax_cli/
│
├── apps/
│   ├── web/
│   └── desktop/
│
├── services/
│   └── api/
│
├── evals/
├── docs/
├── README.md
├── pyproject.toml
└── uv.lock
```

This is the destination, not a requirement to create every directory
immediately.

The project should grow incrementally.

------------------------------------------------------------------------

# Core Runtime

The runtime is the most important component in Parallax.

Conceptually:

``` python
class Agent:
    async def run(self, task: str) -> AgentResult:
        ...
```

The runtime is responsible for:

-   maintaining execution state
-   constructing model requests
-   exposing available tools
-   executing tool calls
-   feeding results back to the model
-   enforcing iteration limits
-   handling failures
-   managing context
-   tracking usage
-   emitting events
-   supporting cancellation
-   deciding when the run terminates
-   producing a final result

A useful mental model is:

``` text
Agent =
    Model
  + State
  + Tools
  + Environment
  + Context Policy
  + Execution Loop
  + Termination Policy
```

------------------------------------------------------------------------

# Agent Execution Model

The fundamental loop is:

``` text
                    ┌──────────────┐
                    │   User Task  │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Build Context   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │      Model      │
                  └────────┬────────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
           Final Response        Tool Calls
                 │                   │
                 │                   ▼
                 │            ┌─────────────┐
                 │            │ Execute     │
                 │            │ Tools       │
                 │            └──────┬──────┘
                 │                   │
                 │                   ▼
                 │            ┌─────────────┐
                 │            │ Tool Result │
                 │            └──────┬──────┘
                 │                   │
                 │                   ▼
                 │            Update State
                 │                   │
                 │                   └──────────┐
                 │                              │
                 └──────────────────────────────┘
                                │
                                ▼
                         Continue / Stop
```

A simplified runtime can eventually look like:

``` python
class Agent:
    async def run(self, task: str):
        state = AgentState(task=task)

        while not self.should_stop(state):
            response = await self.model.generate(
                messages=self.context.build(state),
                tools=self.tools.definitions(),
            )

            state.record(response)

            if response.is_final:
                break

            results = await self.execute_tools(
                response.tool_calls
            )

            state.record(results)

        return state.to_result()
```

------------------------------------------------------------------------

# Agent vs Agent Run

A key design distinction:

### Agent

Reusable configuration:

``` text
model
system prompt
tool registry
context manager
policies
runtime configuration
```

### AgentRun

One execution:

``` text
task
messages
tool calls
tool results
iteration count
usage
events
status
checkpoints
final result
```

Conceptually:

``` text
Agent
 ├── Model
 ├── Tools
 ├── Context Manager
 └── Policies
       │
       ├── Run #1
       ├── Run #2
       └── Run #3
```

This prevents conversation state from being permanently attached to the
agent configuration.

------------------------------------------------------------------------

# Model Layer

Parallax should be model-agnostic.

The runtime should not be tightly coupled to one provider.

A model abstraction should eventually provide:

``` text
generate()
stream()
tool calling
structured output
token usage
context limits
model metadata
```

Conceptually:

``` python
class Model:
    async def generate(
        self,
        messages,
        tools=None,
    ) -> ModelResponse:
        ...
```

Potential providers may include:

-   OpenRouter
-   OpenAI-compatible APIs
-   OpenAI
-   Anthropic
-   local models
-   other compatible providers

The model layer should expose capabilities such as:

``` text
provider
model_id
context_window
supports_tools
supports_streaming
supports_vision
input_cost
output_cost
```

------------------------------------------------------------------------

# Tool System

Tools are the bridge between the model and the environment.

A tool should have:

``` text
name
description
input schema
function
execution policy
result
metadata
```

Example:

``` python
@tool
def read_file(path: str) -> str:
    """Read a file from the repository."""
    ...
```

The runtime should transform this into a provider-compatible schema:

``` json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file from the repository.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        }
      },
      "required": ["path"]
    }
  }
}
```

Tool execution should eventually support:

-   validation
-   timeouts
-   cancellation
-   permissions
-   structured results
-   error reporting
-   execution metadata
-   logging
-   sandboxing

A tool failure should normally become model-visible feedback rather than
crashing the entire agent.

## Initial Coding Tools

``` text
read_file
write_file
edit_file
list_files
search_code
shell
git
run_tests
```

Later:

``` text
inspect_ast
search_symbols
find_references
apply_patch
run_linter
run_formatter
inspect_dependencies
query_docs
```

------------------------------------------------------------------------

# Repository Intelligence

A coding agent needs more than raw text retrieval.

Parallax should eventually understand repositories through multiple
layers.

### Layer 1 --- Filesystem

``` text
directories
files
file metadata
.gitignore
configuration
```

### Layer 2 --- Text Search

``` text
grep-like search
regex search
keyword search
filename search
```

### Layer 3 --- Structural Search

``` text
functions
classes
methods
imports
symbols
references
```

### Layer 4 --- AST

Tree-sitter and similar parsing technologies can provide language-aware
structure.

### Layer 5 --- Repository Graph

``` text
file
 ├── imports
 ├── defines
 ├── references
 ├── tests
 └── dependencies
```

This allows Parallax to reason about a repository structurally instead
of repeatedly reading arbitrary files.

------------------------------------------------------------------------

# Context Engineering

Context is one of the central engineering problems in coding agents.

The runtime cannot simply append everything it sees to the prompt
forever.

Parallax should eventually manage:

-   context budgets
-   message history
-   tool results
-   relevant files
-   repository instructions
-   summaries
-   compaction
-   retrieval
-   prioritization
-   progressive disclosure

The context manager should answer:

> What information does the model need right now to make the next
> decision?

Conceptually:

``` text
Repository
     │
     ├── relevant files
     ├── symbols
     ├── search results
     ├── previous actions
     ├── tool results
     └── task information
              │
              ▼
        Context Manager
              │
              ▼
        Model Context
```

------------------------------------------------------------------------

# RAG

RAG will primarily support repository intelligence and contextual
retrieval.

Potential pipeline:

``` text
Repository
    ↓
Parse / Chunk
    ↓
Metadata
    ↓
Embeddings
    ↓
Vector Index
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
Context Manager
    ↓
LLM
```

Parallax should not depend exclusively on vector search.

Coding repositories benefit from:

-   lexical search
-   symbol search
-   AST-aware search
-   graph relationships
-   vector similarity

Therefore the long-term approach should support **hybrid retrieval**.

Potential technologies:

-   PostgreSQL
-   pgvector
-   local vector stores
-   BM25/lexical retrieval
-   Tree-sitter
-   embedding APIs or local embedding models

------------------------------------------------------------------------

# MCP

Model Context Protocol will be an important interoperability layer.

Parallax should eventually support MCP servers for capabilities such as:

``` text
documentation
databases
browser tools
issue trackers
internal tools
developer services
```

The MCP subsystem should handle:

``` text
server discovery
connection lifecycle
tool discovery
tool execution
resources
prompts
transport
authorization
errors
```

Parallax should use an established MCP implementation rather than
reimplementing the protocol itself.

The runtime's responsibility is to integrate MCP capabilities into its
own tool/context model.

------------------------------------------------------------------------

# Subagents

Complex tasks can eventually be delegated to specialized agents.

Example:

``` text
                    Main Agent
                        │
              ┌─────────┼─────────┐
              │         │         │
              ▼         ▼         ▼
           Research   Coding     Testing
            Agent      Agent      Agent
```

Potential responsibilities:

``` text
Research Agent
 └── investigate repository / documentation

Coding Agent
 └── implement changes

Testing Agent
 └── run and analyze tests

Review Agent
 └── inspect resulting changes
```

The system should support:

-   delegation
-   handoffs
-   isolated state
-   shared context
-   result passing
-   parallel execution
-   cancellation
-   budgets
-   failure handling

Subagents should be introduced only after the single-agent runtime is
stable.

------------------------------------------------------------------------

# Sandboxing and Security

A coding agent can execute arbitrary commands and modify arbitrary
files.

Security is therefore a first-class concern.

Potential controls:

``` text
filesystem restrictions
working-directory restrictions
command allowlists
environment isolation
network restrictions
resource limits
process isolation
secret protection
permission prompts
```

A future sandbox may use containerization or another isolated execution
environment.

Security concerns to address include:

-   prompt injection
-   malicious repository content
-   untrusted MCP servers
-   accidental destructive commands
-   secret exfiltration
-   network access
-   arbitrary code execution
-   tool permission escalation

Tool execution should be treated as a security boundary rather than
assuming model-generated commands are trustworthy.

------------------------------------------------------------------------

# Sessions and Persistence

Sessions should eventually become first-class objects.

A session may contain:

``` text
user
repository
working directory
model
configuration
messages
runs
tool calls
checkpoints
usage
state
```

Conceptual hierarchy:

``` text
User
 └── Session
      ├── Run
      │    ├── Messages
      │    ├── Tool Calls
      │    ├── Events
      │    └── Result
      │
      ├── Run
      └── Run
```

Local mode may initially use:

``` text
SQLite
```

Platform mode may use:

``` text
PostgreSQL
```

The runtime should not require a cloud database to operate locally.

------------------------------------------------------------------------

# Events and Streaming

The runtime should be event-driven.

Possible events:

``` text
run_started
model_started
model_delta
model_completed

tool_started
tool_output
tool_completed
tool_failed

file_changed

test_started
test_completed

agent_message
agent_error

run_completed
run_cancelled
```

Example:

``` python
async for event in agent.run_stream(task):
    ...
```

The CLI could render these events directly.

The desktop application could subscribe through a local process
boundary.

The web application could receive the same event stream through
WebSocket or SSE.

This keeps the runtime independent of presentation.

------------------------------------------------------------------------

# Evals

Parallax should eventually contain its own evaluation system.

For coding agents, useful evaluation signals include:

``` text
task completion
tests passing
files correctly modified
regressions
tool errors
iterations
latency
token usage
cost
human intervention
```

A benchmark task might look like:

``` text
Repository: example-project
Task: Fix issue #42

Expected:
    - tests pass
    - regression test exists
    - unrelated files unchanged
```

The evaluation harness should make it possible to compare:

``` text
model A vs model B
prompt A vs prompt B
runtime version A vs runtime version B
tool strategy A vs tool strategy B
```

without relying only on subjective manual testing.

------------------------------------------------------------------------

# Observability

Agent systems are difficult to debug without structured telemetry.

Parallax should eventually track:

``` text
run duration
model latency
time to first token
input tokens
output tokens
cached tokens
estimated cost
tool latency
tool failures
iterations
context size
subagent activity
MCP activity
```

Tracing should make it possible to understand:

``` text
Run
 ├── Model call
 │    ├── context size
 │    ├── latency
 │    └── usage
 │
 ├── Tool call
 │    ├── command
 │    ├── duration
 │    └── result
 │
 ├── Model call
 │
 └── Final result
```

OpenTelemetry is a possible foundation for standardized tracing.

------------------------------------------------------------------------

# Interfaces

## CLI

The CLI is the first interface.

Initial goal:

``` bash
parallax "Fix the authentication tests"
```

Potential commands:

``` bash
parallax
parallax run "Fix this bug"
parallax inspect
parallax session
parallax config
parallax models
parallax mcp
parallax eval
```

The CLI should remain thin.

It should primarily:

1.  Parse user input.
2.  Configure the runtime.
3.  Subscribe to runtime events.
4.  Render output.
5.  Forward user interaction to the runtime.

------------------------------------------------------------------------

## Desktop

The desktop application is a future interface.

Potential architecture:

``` text
Tauri
 ├── Rust host
 ├── TypeScript/React UI
 └── Parallax Core process
```

Potential capabilities:

-   repository explorer
-   agent chat
-   diff viewer
-   terminal
-   task history
-   session management
-   model selection
-   tool permissions
-   MCP configuration
-   agent activity visualization

The desktop application should consume the same core rather than
implementing another agent runtime.

------------------------------------------------------------------------

## Web

The web application may eventually use:

``` text
Next.js
TypeScript
FastAPI
Parallax Core
PostgreSQL
```

Conceptually:

``` text
Browser
   │
   ▼
Next.js
   │
   ▼
API
   │
   ▼
Parallax Runtime
   │
   ├── Model Providers
   ├── Tools
   ├── MCP
   └── Repository
```

The web application introduces additional platform concerns such as:

-   authentication
-   authorization
-   sessions
-   organizations
-   usage limits
-   model policies
-   cloud persistence
-   job execution
-   billing
-   rate limiting

These concerns should not be embedded into the local core.

------------------------------------------------------------------------

# Platform Layer

The platform layer is optional and sits outside the core runtime.

Potential responsibilities:

``` text
API
Authentication
OAuth / OIDC
Users
Organizations
Permissions
Model access policies
Usage tracking
Quotas
Billing
Cloud persistence
Metrics
Rate limiting
```

For example, a hosted Parallax service could decide:

``` text
User A
 └── can use models X and Y

User B
 └── can use models X, Y and Z

Organization C
 └── has a monthly token quota
```

The core runtime should instead receive the effective configuration and
execute the run.

This separation allows:

``` text
Local Parallax
    ↓
user's API key
user's repository
local execution

Hosted Parallax
    ↓
platform authentication
platform model gateway
cloud persistence
remote execution
```

Both can use the same runtime architecture.

------------------------------------------------------------------------

# Usage and Model Policies

A future platform model gateway can track:

``` text
user
session
run
provider
model
input tokens
output tokens
cached tokens
cost
latency
```

This enables policies such as:

``` text
allowed models
token limits
rate limits
usage quotas
provider restrictions
organization policies
```

The runtime itself should remain capable of running independently.

------------------------------------------------------------------------

# Technology Direction

## Core

Primary language:

``` text
Python
```

Python is the initial choice because of its AI/LLM ecosystem, async
support, rapid iteration, and broad developer tooling.

## CLI

Initial:

``` text
Python
Typer
Rich
```

Potential future native CLI:

``` text
Rust
```

Rust should not be introduced into the runtime prematurely.

A native Rust CLI may eventually make sense for startup performance,
distribution, native system integration, or tighter process management.
Until then, a Python CLI keeps the architecture simpler.

## Desktop

Potential:

``` text
Tauri
Rust
React
TypeScript
```

## Web

Potential:

``` text
Next.js
React
TypeScript
Tailwind CSS
FastAPI
```

## Data

Local:

``` text
SQLite
```

Platform:

``` text
PostgreSQL
```

Potential vector search:

``` text
pgvector
```

Redis should be introduced only when a real workload requires it.

## Infrastructure

Potential:

``` text
Docker
Docker Compose
GitHub Actions
Vercel
Cloud infrastructure as required
```

Infrastructure should not become the focus before the runtime itself is
mature.

------------------------------------------------------------------------

# Design Principles

## 1. Core first

Build the runtime before building multiple UIs.

``` text
Core → CLI → Desktop/Web
```

not:

``` text
CLI + Web + Desktop + Core simultaneously
```

## 2. Interface independence

The core should not contain React, Next.js, terminal formatting, HTTP
route handlers, or browser-specific logic.

It should expose clean APIs and events.

## 3. Model independence

Do not hardcode the runtime around one provider.

## 4. Local-first

The runtime should work with:

``` text
local repository
local filesystem
user-owned API keys
local persistence
```

without requiring a hosted Parallax account.

## 5. Explicit state

Agent state should be inspectable and structured.

Avoid hiding important runtime behavior behind opaque abstractions.

## 6. Async by design

The runtime will eventually need:

``` text
streaming
parallel tools
MCP
subagents
cancellation
concurrent operations
```

Therefore asynchronous execution should be adopted early.

## 7. Events over UI coupling

The runtime emits events. Clients decide how those events are rendered.

## 8. Security by design

Any capability that can execute commands, modify files, access networks,
or read secrets must be treated as privileged.

## 9. Evaluate important behavior

New agent capabilities should be measurable.

## 10. Build understanding before abstractions

Early versions should intentionally implement core mechanics directly.
Frameworks can be evaluated and introduced after the underlying
architecture is understood.

------------------------------------------------------------------------

# Development Roadmap

## Phase 0 --- Foundation

``` text
Python
uv
project structure
environment management
typing
dataclasses
Pydantic
pytest
asyncio
```

## Phase 1 --- Minimal Agent Runtime

Implement:

``` text
LLM request
message state
tool schemas
tool registry
tool execution
tool results
iteration loop
termination
```

Initial runtime:

``` text
LLM
 ↓
Tool Call
 ↓
Tool
 ↓
Result
 ↓
LLM
```

## Phase 2 --- Runtime Robustness

Add:

``` text
async execution
max iterations
timeouts
tool errors
structured ToolResult
retry handling
Agent
AgentRun
cancellation
```

## Phase 3 --- Coding Agent

Add:

``` text
read_file
write_file
edit_file
list_files
search_code
shell
git
test runner
```

## Phase 4 --- Events and Streaming

Add:

``` text
AgentEvent
streaming model output
tool lifecycle events
file change events
execution timing
usage tracking
structured logs
```

## Phase 5 --- Context Engineering

Add:

``` text
context manager
token budgets
history management
repository instructions
context compaction
summarization
progressive disclosure
```

## Phase 6 --- Repository Intelligence

Add:

``` text
Tree-sitter
symbol indexing
AST analysis
reference search
repository graph
hybrid retrieval
```

## Phase 7 --- RAG

Add:

``` text
embeddings
vector search
lexical retrieval
hybrid retrieval
reranking
repository indexing
```

## Phase 8 --- MCP

Add:

``` text
MCP client
server configuration
tool discovery
resource discovery
tool execution
permissions
```

## Phase 9 --- Subagents

Add:

``` text
delegation
agent roles
parallel agents
handoffs
shared state
subagent budgets
```

## Phase 10 --- Sandboxing

Add:

``` text
process isolation
filesystem isolation
network policy
resource limits
permission model
secret protection
```

## Phase 11 --- Persistence

Add:

``` text
sessions
runs
checkpoints
SQLite
run recovery
history
```

## Phase 12 --- Evals

Add:

``` text
benchmark tasks
automated graders
regression suite
runtime comparisons
model comparisons
tool effectiveness measurements
```

## Phase 13 --- Observability

Add:

``` text
structured logging
metrics
tracing
token accounting
cost tracking
latency
tool analytics
```

## Phase 14 --- Interfaces

Build:

``` text
CLI
 ↓
Desktop
 ↓
Web
```

while keeping the core unchanged.

## Phase 15 --- Platform

Eventually introduce:

``` text
FastAPI
authentication
OAuth/OIDC
users
organizations
model gateway
usage policies
quotas
cloud persistence
billing
```

This phase should happen only after the core runtime is mature.

------------------------------------------------------------------------

# Current Implementation

The initial prototype demonstrates:

``` text
OpenRouter model call
        ↓
tool schema generation
        ↓
tool registration
        ↓
LLM tool calling
        ↓
tool execution
        ↓
tool result
        ↓
LLM continuation
```

The prototype uses Python and an OpenRouter client.

The current code is a learning implementation and is expected to be
refactored into the `parallax-core` package.

Important early improvements include:

``` text
fix tool registration
fix provider-compatible message serialization
introduce a proper Tool abstraction
introduce model abstraction
make runtime asynchronous
add max iterations
add tool error handling
add structured ToolResult
separate Agent from AgentRun
```

The prototype should not be discarded. It is the seed from which the
runtime is being built.

------------------------------------------------------------------------

# Recommended Initial Core API

A long-term API may look approximately like:

``` python
from parallax import Agent

agent = Agent(
    model="provider/model",
    tools="auto",
)

result = await agent.run(
    "Fix the failing authentication tests."
)

print(result.output)
```

For streaming:

``` python
async for event in agent.run_stream(
    "Fix the failing authentication tests."
):
    print(event)
```

The exact API is expected to evolve as the runtime becomes more capable.

------------------------------------------------------------------------

# Learning Objectives

Parallax is deliberately designed to develop practical AI engineering
skills.

### Python systems engineering

``` text
typing
dataclasses
Pydantic
asyncio
subprocess
filesystem APIs
packaging
testing
logging
```

### LLM engineering

``` text
messages
tool calling
structured output
streaming
context windows
tokens
model capabilities
provider abstraction
```

### Agent engineering

``` text
agent loops
state machines
tool execution
termination
retries
context management
planning
delegation
subagents
```

### Coding-agent engineering

``` text
Git
patches
diffs
repository search
ASTs
tests
build systems
code indexing
```

### Retrieval

``` text
embeddings
vector search
BM25
hybrid retrieval
reranking
RAG
```

### Systems engineering

``` text
processes
sandboxing
IPC
concurrency
cancellation
persistence
event systems
```

### Backend engineering

Later:

``` text
REST
WebSockets
SSE
authentication
OAuth
sessions
databases
rate limiting
quotas
```

### Production AI engineering

``` text
observability
tracing
evaluations
cost tracking
model routing
security
reliability
```

------------------------------------------------------------------------

# What Parallax Will Not Reinvent

The project should build important AI-agent abstractions itself for
learning purposes, but should not waste time reimplementing mature
foundational technology.

## Build in-house

``` text
agent execution loop
agent state
tool registry abstraction
runtime policies
context management
event system
agent orchestration
subagent orchestration
usage accounting
run persistence
evaluation harness
model abstraction
```

## Use mature libraries/protocols

``` text
HTTP
TLS
cryptography
Git
Tree-sitter
Docker
MCP protocol
OAuth/OIDC
database drivers
tokenizers
LLM inference
provider SDKs
```

The objective is to understand the architecture without unnecessarily
rebuilding foundational infrastructure.

------------------------------------------------------------------------

# Long-Term Vision

``` text
                         PARALLAX
                            │
             ┌──────────────┴──────────────┐
             │                             │
      Parallax Core                 Parallax Platform
             │                             │
       Agent Runtime                 API / Auth
       Model Layer                  Users / Orgs
       Tool System                 Usage / Quotas
       Context Engine              Model Gateway
       Repository Intel            Cloud Persistence
       MCP                          Billing
       Subagents                   Metrics
       Sandbox
       Persistence
       Evals
       Observability
             │
     ┌───────┼────────┐
     │       │        │
    CLI   Desktop     Web
```

The final system should make it possible to use the same agent runtime
in multiple environments:

``` text
Local developer
    ↓
Parallax CLI
    ↓
Parallax Core
```

or:

``` text
Developer
    ↓
Parallax Desktop
    ↓
Parallax Core
```

or:

``` text
Developer
    ↓
Parallax Web
    ↓
Parallax API
    ↓
Parallax Core
```

The runtime remains the common foundation.

------------------------------------------------------------------------

# Project Philosophy

Parallax is not being built simply to produce another chatbot or another
wrapper around an LLM API.

The project is intended to answer a deeper engineering question:

> **What does it take to build a robust, model-agnostic AI agent runtime
> capable of operating inside real software repositories?**

That means understanding and implementing the systems around the model:

``` text
Models
   +
Tools
   +
State
   +
Context
   +
Environment
   +
Repository Intelligence
   +
Memory
   +
MCP
   +
Subagents
   +
Security
   +
Evaluation
   +
Observability
```

The LLM is one component of the system.

**Parallax is the system around it.**

------------------------------------------------------------------------

## Status

🚧 **Early development**

The project is currently transitioning from a minimal LLM + tool-calling
prototype into a reusable `parallax-core` runtime.

The immediate objective is to establish a clean, asynchronous,
model-agnostic runtime before adding advanced capabilities such as
repository indexing, RAG, MCP, subagents, sandboxing, and platform
services.

------------------------------------------------------------------------

## License

License to be decided.