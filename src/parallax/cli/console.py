from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from parallax.cli.theme import (
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


def _get_git_status() -> str:
    """Get active git branch and working tree cleanliness."""
    if shutil.which("git"):
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            cleanliness = "dirty" if status else "clean"
            return f"{branch}:{cleanliness}"
        except Exception:
            pass
    return "non-git"


def print_hero(
    model: str,
    workspace: str | Path | None = None,
    version: str = "0.1.2",
    console: Console | None = None,
) -> None:
    """Render the flagship Parallax landing card with text-based branding and shortcuts."""
    console = console or get_console()
    ws_str = str(workspace or Path.cwd())
    home = str(Path.home())
    if ws_str.startswith(home):
        ws_str = "~" + ws_str[len(home) :]

    git_info = _get_git_status()

    # Brand header
    brand = Text()
    brand.append("PARALLAX", style="bold white")
    brand.append(f"  v{version}\n", style="dim #818cf8")
    brand.append("autonomous coding runtime", style="dim #94a3b8")

    # Session metadata table
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="muted", width=11, no_wrap=True)
    meta.add_column(style="tool.name")

    meta.add_row("model", f"{model}")
    meta.add_row("workspace", f"{ws_str} [dim #64748b]({git_info})[/dim #64748b]")

    # Quick Commands table
    cmds = Table.grid(padding=(0, 2))
    cmds.add_column(style="brand", width=12, no_wrap=True)
    cmds.add_column(style="dim #94a3b8")

    cmds.add_row("/help", "interactive guide & keyboard shortcuts")
    cmds.add_row("/model", "switch active LLM model")
    cmds.add_row("/context", "token usage & turn count")
    cmds.add_row("/doctor", "system health & environment check")
    cmds.add_row("/clear", "clear terminal screen & session history")

    # Combine inside a structured vertical card
    main = Table.grid(padding=(1, 0))
    main.add_column()
    main.add_row(brand)
    main.add_row(meta)
    main.add_row(Text("Quick Commands", style="bold #818cf8"))
    main.add_row(cmds)
    main.add_row(Text("Ctrl+C interrupt · Ctrl+D exit · / for command suggestions", style="dim #64748b"))

    panel = Panel(
        main,
        box=PANEL_BOX,
        border_style="#334155",
        padding=(1, 3),
        title=f"[brand]parallax[/brand] [dim #64748b]v{version}[/dim #64748b]",
        title_align="left",
    )

    console.print()
    console.print(panel)
    console.print("[dim #64748b]Type instructions to begin, or / for command suggestions[/dim #64748b]")
    console.print()


def print_rule(title: str = "", style: str = "border") -> None:
    """Print a subtle horizontal divider line."""
    console = get_console()
    console.rule(title, style=style)
