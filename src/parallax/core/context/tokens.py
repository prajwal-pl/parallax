from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parallax.core.types import Message


class TokenEstimator:
    """
    Estimates token counts. Uses tiktoken when its vocabulary is available and
    falls back to a character heuristic otherwise (e.g. offline, where tiktoken
    cannot download its vocabulary).
    """

    def __init__(self) -> None:
        self._enc = None
        self._loaded = False  # load the vocabulary lazily; it is slow/needs network

    def _encoding(self):
        if not self._loaded:
            self._loaded = True
            try:
                import tiktoken

                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None
        return self._enc

    def count_text(self, text: str | None) -> int:
        if not text:
            return 0
        enc = self._encoding()
        if enc is not None:
            return len(enc.encode(text, disallowed_special=()))
        return math.ceil(len(text) / 3.7)

    def count_message(self, message: Message) -> int:
        tokens = 4  # per-message framing overhead
        tokens += self.count_text(message.content)
        for tc in message.tool_calls or []:
            tokens += self.count_text(tc.name)
            tokens += self.count_text(json.dumps(tc.arguments))
        return tokens

    def count_messages(self, messages: list[Message]) -> int:
        return sum(self.count_message(m) for m in messages) + 2  # reply priming
