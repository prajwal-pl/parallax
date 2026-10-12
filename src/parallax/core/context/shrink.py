from __future__ import annotations

import re

from parallax.core.context.tokens import TokenEstimator
from parallax.core.types import Message

_LIMIT_RE = re.compile(r"limit\s+(\d+)", re.IGNORECASE)
_STEPS = (1500, 600, 250, 100)  # progressively tighter per-tool-output char caps


def is_request_too_large(exc: BaseException) -> bool:
    """Provider rejected the request because it exceeds a per-request/minute token cap (HTTP 413)."""
    if getattr(exc, "status_code", None) == 413:
        return True
    msg = str(exc).lower()
    return "request too large" in msg or "error code: 413" in msg


def extract_token_limit(exc: BaseException) -> int | None:
    m = _LIMIT_RE.search(str(exc))
    return int(m.group(1)) if m else None


def _clip(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head = int(cap * 0.7)
    tail = cap - head
    omitted = len(text) - cap
    return f"{text[:head]}\n... [{omitted} chars omitted to fit provider token limit] ...\n{text[-tail:] if tail else ''}"


def shrink_to_fit(
    messages: list[Message],
    budget_tokens: int,
    estimator: TokenEstimator | None = None,
    reserved_tokens: int = 0,
) -> bool:
    """
    Shrink tool outputs in place (oldest first) until the estimated prompt fits
    `budget_tokens - reserved_tokens`. Returns True if it now fits.
    The system prompt and user/assistant text are never altered.
    """
    est = estimator or TokenEstimator()
    target = max(budget_tokens - reserved_tokens, 200)

    def total() -> int:
        return est.count_messages(messages)

    for cap in _STEPS:
        if total() <= target:
            return True
        for m in messages:
            if m.role == "tool" and len(m.content) > cap:
                m.content = _clip(m.content, cap)
                if total() <= target:
                    return True
    return total() <= target
