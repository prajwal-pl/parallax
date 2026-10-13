from collections.abc import AsyncIterator, Callable
from typing import Any
import os

from parallax.core.context import ContextConfig, ContextManager
from parallax.core.errors import DuplicateToolError
from parallax.core.llm import LLMClient, OpenRouterClient, GroqClient, RetryPolicy
from parallax.core.run import AgentRun
from parallax.core.tool import extract_schema
from parallax.core.types import AgentEvent, AgentResult, Message



KNOWN_GROQ_MODELS: set[str] = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
}


class Agent:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        llm: LLMClient | None = None,
        retry: RetryPolicy | None = None,
        context: ContextConfig | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        if llm is not None:
            self.llm = llm
        else:
            provider = os.getenv("PARALLAX_PROVIDER", "").lower()
            use_groq = (
                provider == "groq"
                or model.startswith("groq/")
                or (provider != "openrouter" and model in KNOWN_GROQ_MODELS)
            )
            if use_groq:
                self.llm = GroqClient()
                if model.startswith("groq/"):
                    self.model = model[len("groq/"):]
            else:
                self.llm = OpenRouterClient()
        self.retry = retry or RetryPolicy()
        self.context = ContextManager(context)
        self.tool_registry: dict[str, Callable[..., Any]] = {}
        self.tool_schemas: list[dict[str, Any]] = []
        
    @property
    def provider_name(self) -> str:
        """Name of the active LLM provider."""
        return "Groq" if isinstance(self.llm, GroqClient) else "OpenRouter"

    @property
    def workspace(self) -> str:
        """Configured workspace directory for the agent."""
        return self.context.config.workspace

    def set_model(self, model: str, provider: str | None = None) -> None:
        """Update active model and configure LLM provider client accordingly."""
        if provider in ("GroqClient", "Groq"):
            self.model = model[len("groq/"):] if model.startswith("groq/") else model
            if not isinstance(self.llm, GroqClient):
                self.llm = GroqClient()
            return
        if provider in ("OpenRouterClient", "OpenRouter"):
            self.model = model
            if not isinstance(self.llm, OpenRouterClient):
                self.llm = OpenRouterClient()
            return

        if model.startswith("groq/"):
            self.model = model[len("groq/"):]
            if not isinstance(self.llm, GroqClient):
                self.llm = GroqClient()
        elif model in KNOWN_GROQ_MODELS:
            self.model = model
            if not isinstance(self.llm, GroqClient):
                self.llm = GroqClient()
        elif "/" not in model and isinstance(self.llm, GroqClient):
            self.model = model
        else:
            self.model = model
            if not isinstance(self.llm, OpenRouterClient):
                self.llm = OpenRouterClient()

    def add_tool(self, func: Callable[..., Any]) -> None:
        name = func.__name__
        if name in self.tool_registry:
            raise DuplicateToolError(f"Tool '{name}' is already registered")
        schema = extract_schema(func)
        self.tool_registry[name] = func
        self.tool_schemas.append(schema.to_dict())

    async def run(
        self, user_input: str, history: list[Message] | None = None
    ) -> AgentResult:
        state = AgentRun(self, user_input, history=history)
        return await state.execute()

    async def run_stream(
        self, user_input: str, history: list[Message] | None = None
    ) -> AsyncIterator[AgentEvent]:
        """
        Stream lifecycle events of a run in real time.

        Failures are reported as a RUN_FAILED event; unlike `run()`, the
        underlying exception is not re-raised to the consumer.
        """
        state = AgentRun(self, user_input, history=history)
        async for event in state.stream():
            yield event

