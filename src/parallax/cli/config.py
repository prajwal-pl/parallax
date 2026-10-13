from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from parallax.cli.console import get_console
from parallax.cli.theme import ICON_ERROR, ICON_SUCCESS, ICON_USER, ICON_WARNING, PANEL_BOX


def get_config_dir() -> Path:
    """Return the global Parallax user configuration directory ~/.parallax"""
    config_dir = Path.home() / ".parallax"
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Return the global Parallax configuration file path ~/.parallax/config.json"""
    return get_config_dir() / "config.json"


def load_global_config() -> dict[str, str]:
    """
    Load environment variables from ~/.parallax/config.json and ~/.parallax/.env.
    Only populates os.environ if the variable is not already defined in the active process.
    """
    config_data: dict[str, str] = {}
    cfg_file = get_config_file()

    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    config_data.update({str(k): str(v) for k, v in loaded.items() if v})
        except Exception:
            pass

    # Also check ~/.parallax/.env for users who prefer standard env syntax
    global_env = get_config_dir() / ".env"
    if global_env.is_file():
        try:
            from dotenv import dotenv_values
            env_vals = dotenv_values(global_env)
            for k, v in env_vals.items():
                if k and v and k not in config_data:
                    config_data[k] = v
        except Exception:
            pass

    # Inject into os.environ without overwriting active environment
    for key, value in config_data.items():
        if key not in os.environ and value:
            os.environ[key] = value

    return config_data


def save_global_config(updates: dict[str, str]) -> Path:
    """
    Save key-value pairs into ~/.parallax/config.json with restrictive file permissions.
    """
    cfg_file = get_config_file()
    existing: dict[str, str] = {}

    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, dict):
                    existing = content
        except Exception:
            existing = {}

    existing.update(updates)

    # Write atomically with 0o600 permissions
    temp_file = cfg_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    os.chmod(temp_file, 0o600)
    temp_file.replace(cfg_file)

    # Ensure updated keys are active in the running process
    for k, v in updates.items():
        os.environ[k] = v

    return cfg_file


def has_configured_keys() -> bool:
    """Check if any supported model provider API key is available."""
    return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def prompt_onboarding(console: Console | None = None) -> bool:
    """
    Interactive first-run wizard to configure an LLM provider key if none is present.
    Returns True if an API key was configured, False otherwise.
    """
    console = console or get_console()

    # Do not prompt in non-interactive / piped environments
    if not sys.stdin.isatty():
        return False

    intro = Text()
    intro.append("Welcome to Parallax!\n\n", style="bold #818cf8")
    intro.append(
        "Parallax is a model-agnostic AI coding agent runtime.\n"
        "To start reasoning and generating code, Parallax needs an API key to connect to an LLM provider.\n\n",
        style="white",
    )
    intro.append("Choose an inference provider:\n", style="bold #94a3b8")
    intro.append("  [1] Groq ", style="bold #34d399")
    intro.append("(Recommended · Ultra-fast & Free tier available at https://console.groq.com)\n", style="muted")
    intro.append("  [2] OpenRouter ", style="bold #38bdf8")
    intro.append("(200+ models from OpenAI, Anthropic, Meta at https://openrouter.ai)\n", style="muted")
    intro.append("  [3] Skip for now ", style="dim")
    intro.append("(You can configure keys later via .env or ~/.parallax/config.json)\n", style="muted")

    panel = Panel(
        intro,
        title="[brand]parallax setup[/brand] [muted]· first-run configuration[/muted]",
        title_align="left",
        box=PANEL_BOX,
        border_style="#334155",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()

    choice = console.input("[brand]› Select provider [1/2/3] (default: 1):[/brand] ").strip() or "1"

    if choice == "1":
        key = console.input("[brand]› Enter your Groq API Key (starts with gsk_):[/brand] ").strip()
        if key:
            save_global_config({"GROQ_API_KEY": key})
            console.print(
                f"\n  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]Key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]. "
                f"[muted]Defaulting to[/muted] [tool.name]llama-3.3-70b-versatile[/tool.name].\n"
            )
            return True
        else:
            console.print(f"\n  [status.warning]{ICON_WARNING}[/status.warning] [muted]No key entered. Continuing without credentials.[/muted]\n")
            return False

    elif choice == "2":
        key = console.input("[brand]› Enter your OpenRouter API Key (starts with sk-or-):[/brand] ").strip()
        if key:
            save_global_config({"OPENROUTER_API_KEY": key})
            console.print(
                f"\n  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]Key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]. "
                f"[muted]Defaulting to[/muted] [tool.name]liquid/lfm-2.5-2.6b:free[/tool.name].\n"
            )
            return True
        else:
            console.print(f"\n  [status.warning]{ICON_WARNING}[/status.warning] [muted]No key entered. Continuing without credentials.[/muted]\n")
            return False

    else:
        console.print(f"\n  [muted]Setup skipped. You can configure keys in a [tool.name].env[/tool.name] file or run [tool.name]parallax doctor[/tool.name].[/muted]\n")
        return False
