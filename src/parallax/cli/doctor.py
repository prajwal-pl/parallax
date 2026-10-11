from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.text import Text

from parallax.cli.console import get_console
from parallax.cli.theme import (
    ICON_BULLET,
    ICON_ERROR,
    ICON_SUCCESS,
    ICON_WARNING,
    PANEL_BOX,
)


def run_doctor(console: Console | None = None) -> bool:
    """
    Run environment and health diagnostics for Parallax.
    Returns True if critical checks pass, False otherwise.
    """
    console = console or get_console()
    all_healthy = True

    table = Table(
        title="[brand]parallax doctor[/brand] [muted]· system diagnostics[/muted]",
        title_justify="left",
        box=PANEL_BOX,
        border_style="border",
        show_header=True,
        header_style="bold #818cf8",
        padding=(0, 1),
    )
    table.add_column("check", style="tool.name", no_wrap=True)
    table.add_column("status", style="status.info", no_wrap=True)
    table.add_column("detail", style="muted")

    # 1. Python version check
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        table.add_row(
            "Python",
            f"[status.success]{ICON_SUCCESS} ok[/status.success]",
            f"Python {py_version} ({sys.executable})",
        )
    else:
        all_healthy = False
        table.add_row(
            "Python",
            f"[status.error]{ICON_ERROR} outdated[/status.error]",
            f"Python {py_version} (requires Python 3.11+)",
        )

    # 2. OpenRouter API Key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        masked = f"{openrouter_key[:4]}...{openrouter_key[-4:]}" if len(openrouter_key) > 8 else "***"
        table.add_row(
            "OPENROUTER_API_KEY",
            f"[status.success]{ICON_SUCCESS} set[/status.success]",
            f"Configured ({masked})",
        )
    else:
        all_healthy = False
        table.add_row(
            "OPENROUTER_API_KEY",
            f"[status.error]{ICON_ERROR} missing[/status.error]",
            "Required for model inference. Set in .env or environment.",
        )

    # 3. Tavily API Key (Optional)
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        table.add_row(
            "TAVILY_API_KEY",
            f"[status.success]{ICON_SUCCESS} set[/status.success]",
            "Configured for search capability",
        )
    else:
        table.add_row(
            "TAVILY_API_KEY",
            f"[status.warning]{ICON_WARNING} optional[/status.warning]",
            "Optional: enables live web search tool if provided",
        )

    # 4. Git binary and repo check
    git_bin = shutil.which("git")
    if git_bin:
        try:
            branch_out = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            table.add_row(
                "Git Repository",
                f"[status.success]{ICON_SUCCESS} repo[/status.success]",
                f"Branch: {branch_out} (git: {git_bin})",
            )
        except Exception:
            table.add_row(
                "Git Repository",
                f"[status.warning]{ICON_WARNING} non-git[/status.warning]",
                "Working directory is not inside a git repository",
            )
    else:
        table.add_row(
            "Git Binary",
            f"[status.error]{ICON_ERROR} missing[/status.error]",
            "Git binary not found in PATH",
        )

    # 5. Workspace write permissions
    cwd = Path.cwd()
    can_write = os.access(cwd, os.W_OK)
    if can_write:
        table.add_row(
            "Workspace",
            f"[status.success]{ICON_SUCCESS} writable[/status.success]",
            str(cwd),
        )
    else:
        all_healthy = False
        table.add_row(
            "Workspace",
            f"[status.error]{ICON_ERROR} read-only[/status.error]",
            f"Cannot write to {cwd}",
        )

    console.print()
    console.print(table)
    console.print()

    return all_healthy
