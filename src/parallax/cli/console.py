from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
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


def _detect_provider(model: str) -> str:
    if not model:
        return ""
    m = model.lower()
    if m.startswith("groq/") or m in (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ):
        return "Groq"
    elif m.startswith("openrouter/") or "/" in m:
        return "OpenRouter"
    return "Groq"


def _build_wordmark() -> Text:
    t = Text()
    t.append("█▀█ ▄▀█ █▀█ ▄▀█ █   █   ▄▀█ ▀▄▀\n", style="bold #e4e4e7")
    t.append("█▀▀ █▀█ █▀▄ █▀█ █▄▄ █▄▄ █▀█ █ █", style="bold #a1a1aa")
    return t


def print_hero(
    model: str,
    workspace: str | Path | None = None,
    session_id: str | None = None,
    version: str = "0.1.4",
    console: Console | None = None,
) -> None:
    """Render the flagship Parallax landing card with minimalist titanium styling."""
    console = console or get_console()
    ws_str = str(workspace or Path.cwd())
    home = str(Path.home())
    if ws_str.startswith(home):
        ws_str = "~" + ws_str[len(home) :]

    git_info = _get_git_status()
    provider = _detect_provider(model)

    # Stylized brand header
    header = Table.grid(padding=(0, 0))
    header.add_column()
    header.add_row(_build_wordmark())

    sub = Text()
    sub.append(f" v{version} ", style="bold #f4f4f5 bg:#27272a")
    sub.append("  ")
    sub.append("autonomous coding agent runtime", style="italic #71717a")
    header.add_row(sub)

    # System Status & Session HUD
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim #71717a", width=11, no_wrap=True)
    meta.add_column(style="bold #f4f4f5")

    prov_tag = f" [dim #71717a]({provider})[/dim #71717a]" if provider else ""
    meta.add_row("model", f"{model}{prov_tag}")

    git_tag = f" [dim #71717a]({git_info})[/dim #71717a]" if git_info != "non-git" else ""
    meta.add_row("workspace", f"{ws_str}{git_tag}")

    if session_id:
        meta.add_row("session", f"{session_id}  [status.success]● active[/status.success]")
    else:
        meta.add_row("status", "[status.success]● ready[/status.success] [dim #71717a](tools active)[/dim #71717a]")

    # Quick Commands table
    cmds = Table.grid(padding=(0, 2))
    cmds.add_column(style="bold #e4e4e7", width=14, no_wrap=True)
    cmds.add_column(style="#a1a1aa")

    cmds.add_row("/model", "switch active LLM model or provider")
    cmds.add_row("/sessions", "browse, resume, or switch saved sessions")
    cmds.add_row("/context", "view token window usage & turn count")
    cmds.add_row("/doctor", "system health, API keys & environment check")
    cmds.add_row("/clear", "clear terminal screen & start fresh session")
    cmds.add_row("/help", "interactive guide & keyboard shortcuts")

    # Combine inside a structured vertical card
    card = Table.grid(padding=(1, 0), expand=True)
    card.add_column()
    card.add_row(header)
    card.add_row(Rule(style="#27272a"))
    card.add_row(meta)
    card.add_row(Rule(style="#27272a"))
    card.add_row(Text("Quick Commands", style="bold #f4f4f5"))
    card.add_row(cmds)
    card.add_row(Rule(style="#27272a"))
    card.add_row(Text("Ctrl+C interrupt · Ctrl+D exit · / for command suggestions", style="dim #71717a"))

    panel = Panel(
        card,
        box=PANEL_BOX,
        border_style="#3f3f46",
        padding=(1, 3),
        title="[bold #e4e4e7]PARALLAX[/bold #e4e4e7] [dim #71717a]· autonomous runtime[/dim #71717a]",
        title_align="left",
    )

    console.print()
    console.print(panel)
    console.print("[dim #71717a]Type instructions to begin, or[/dim #71717a] [bold #e4e4e7]/[/bold #e4e4e7] [dim #71717a]for command suggestions[/dim #71717a]")
    console.print()


def print_rule(title: str = "", style: str = "border") -> None:
    """Print a subtle horizontal divider line."""
    console = get_console()
    console.rule(title, style=style)
