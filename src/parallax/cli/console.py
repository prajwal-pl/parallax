from __future__ import annotations

import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from parallax.cli.theme import (
    ICON_BULLET,
    PANEL_BOX,
    PARALLAX_THEME,
)

# Shared console singleton
_console: Console | None = None


def get_console() -> Console:
    """Return the global Rich console singleton configured with Parallax theme."""
    global _console
    if _console is None:
        _console = Console(theme=PARALLAX_THEME, highlight=False)
    return _console


def print_hero(model: str, workspace: str | Path | None = None, version: str = "0.1.0") -> None:
    """Render a clean, modern, minimalist header card."""
    console = get_console()
    ws_str = str(workspace or Path.cwd())
    # Shorten user home directory for cleaner appearance
    home = str(Path.home())
    if ws_str.startswith(home):
        ws_str = "~" + ws_str[len(home) :]

    content = Text()
    content.append("model: ", style="muted")
    content.append(model, style="tool.name")
    content.append(f"  {ICON_BULLET}  ", style="muted")
    content.append("workspace: ", style="muted")
    content.append(ws_str, style="tool.name")
    content.append("\n")
    content.append("commands: ", style="muted")
    content.append("/help", style="brand")
    content.append(" · ", style="muted")
    content.append("/model", style="brand")
    content.append(" · ", style="muted")
    content.append("/context", style="brand")
    content.append(" · ", style="muted")
    content.append("/tools", style="brand")
    content.append(" · ", style="muted")
    content.append("/clear", style="brand")
    content.append(" · ", style="muted")
    content.append("Ctrl+D to exit", style="muted")

    panel = Panel(
        content,
        title=f"[brand]parallax[/brand] [muted]v{version}[/muted]",
        title_align="left",
        box=PANEL_BOX,
        border_style="border",
        padding=(0, 1),
    )
    console.print(panel)


def print_rule(title: str = "", style: str = "border") -> None:
    """Print a subtle horizontal divider line."""
    console = get_console()
    console.rule(title, style=style)
