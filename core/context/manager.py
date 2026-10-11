from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.context.compactor import compact
from core.context.instructions import load_instructions
from core.context.tokens import TokenEstimator
from core.context.truncation import truncate_output
from core.types import Message


@dataclass
class ContextConfig:
    workspace: str = "."
    max_context_tokens: int = 32_000
    compaction_threshold: float = 0.8
    keep_recent_turns: int = 2
    max_tool_chars: int = 12_000
    summary_cap_chars: int = 3_000
    load_instructions: bool = True


@dataclass
class UsageBaseline:
    """Provider-reported prompt size for the first `message_count` messages."""

    tokens: int
    message_count: int


class ContextManager:
    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()
        self.estimator = TokenEstimator()

    def build_system_prompt(self, base: str) -> str:
        if not self.config.load_instructions:
            return base
        instructions = load_instructions(self.config.workspace)
        return f"{base}\n\n{instructions}" if instructions else base

    def truncate_tool_output(self, text: str) -> str:
        return truncate_output(text, self.config.max_tool_chars)

    def current_tokens(
        self, messages: list[Message], baseline: UsageBaseline | None = None
    ) -> int:
        """Real provider count for what was already sent + estimate for the rest."""
        if baseline is None or baseline.message_count > len(messages):
            return self.estimator.count_messages(messages)
        newer = messages[baseline.message_count :]
        return baseline.tokens + sum(self.estimator.count_message(m) for m in newer)

    def maybe_compact(
        self, messages: list[Message], baseline: UsageBaseline | None = None
    ) -> tuple[list[Message], dict[str, Any] | None]:
        cfg = self.config
        before = self.current_tokens(messages, baseline)
        if before < cfg.max_context_tokens * cfg.compaction_threshold:
            return messages, None

        compacted, summarized = compact(
            messages, cfg.keep_recent_turns, cfg.summary_cap_chars
        )
        if summarized == 0:
            return messages, None

        after = self.estimator.count_messages(compacted)
        return compacted, {
            "original_tokens": before,
            "compacted_tokens": after,
            "turns_summarized": summarized,
            "still_over_budget": after >= cfg.max_context_tokens,
        }
