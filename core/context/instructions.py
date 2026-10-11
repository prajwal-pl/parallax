from __future__ import annotations

from pathlib import Path

CANDIDATES = ("AGENTS.md", "CLAUDE.md", ".parallax/instructions.md")
MAX_CHARS = 8_000


def load_instructions(workspace: str | Path) -> str | None:
    """Return the first non-empty repo instruction file as a prompt section."""
    root = Path(workspace)
    for name in CANDIDATES:
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            return f"## Repository instructions ({name})\n{text[:MAX_CHARS]}"
    return None
