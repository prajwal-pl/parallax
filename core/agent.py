from collections.abc import AsyncIterator, Callable
from typing import Any

from core.errors import DuplicateToolError
from core.llm import LLMClient, OpenRouterClient, RetryPolicy
from core.run import AgentRun
from core.tool import extract_schema
from core.types import AgentEvent, AgentResult


class Agent:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        llm: LLMClient | None = None,
        retry: RetryPolicy | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.llm: LLMClient = llm or OpenRouterClient()
        self.retry = retry or RetryPolicy()
        self.tool_registry: dict[str, Callable[..., Any]] = {}
        self.tool_schemas: list[dict[str, Any]] = []

    def add_tool(self, func: Callable[..., Any]) -> None:
        name = func.__name__
        if name in self.tool_registry:
            raise DuplicateToolError(f"Tool '{name}' is already registered")
        schema = extract_schema(func)
        self.tool_registry[name] = func
        self.tool_schemas.append(schema.to_dict())

    async def run(self, user_input: str) -> AgentResult:
        state = AgentRun(self, user_input)
        return await state.execute()

    async def run_stream(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """
        Stream lifecycle events of a run in real time.

        Failures are reported as a RUN_FAILED event; unlike `run()`, the
        underlying exception is not re-raised to the consumer.
        """
        state = AgentRun(self, user_input)
        async for event in state.stream():
            yield event
