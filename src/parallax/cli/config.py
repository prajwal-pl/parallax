from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
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


def get_provider_status() -> dict[str, dict[str, Any]]:
    """Return current configuration status and masked keys for supported LLM providers."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    or_key = os.getenv("OPENROUTER_API_KEY", "")

    def mask(k: str) -> str:
        if not k:
            return "—"
        return f"{k[:4]}...{k[-4:]}" if len(k) > 8 else "***"

    return {
        "groq": {
            "name": "Groq",
            "env_var": "GROQ_API_KEY",
            "configured": bool(groq_key),
            "key": groq_key,
            "masked": mask(groq_key),
            "url": "https://console.groq.com",
            "description": "Ultra-fast inference & generous free tier",
            "default_model": "llama-3.3-70b-versatile",
        },
        "openrouter": {
            "name": "OpenRouter",
            "env_var": "OPENROUTER_API_KEY",
            "configured": bool(or_key),
            "key": or_key,
            "masked": mask(or_key),
            "url": "https://openrouter.ai",
            "description": "200+ models from OpenAI, Anthropic, Meta",
            "default_model": "liquid/lfm-2.5-2.6b:free",
        },
    }


def set_provider_key(provider: str, key: str) -> bool:
    """Save an LLM provider key into global config and active os.environ."""
    norm = provider.strip().lower()
    if norm in ("groq", "g"):
        save_global_config({"GROQ_API_KEY": key.strip()})
        return True
    elif norm in ("openrouter", "or"):
        save_global_config({"OPENROUTER_API_KEY": key.strip()})
        return True
    return False


def has_configured_keys() -> bool:
    """Check if any supported model provider API key is available."""
    return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def prompt_onboarding(console: Console | None = None) -> bool:
    """
    Interactive first-run wizard to configure LLM provider keys.
    Allows configuring Groq, OpenRouter, or both.
    Returns True if at least one API key was configured, False otherwise.
    """
    console = console or get_console()

    # Do not prompt in non-interactive / piped environments
    if not sys.stdin.isatty():
        return False

    intro = Text()
    intro.append("Welcome to Parallax!\n\n", style="bold #f4f4f5")
    intro.append(
        "Parallax is a model-agnostic AI coding agent runtime.\n"
        "To start reasoning and generating code, Parallax needs an API key to connect to an LLM provider.\n\n",
        style="white",
    )
    intro.append("Choose an inference provider setup:\n", style="bold #a1a1aa")
    intro.append("  [1] Groq ", style="bold #22c55e")
    intro.append("(Recommended · Ultra-fast & Free tier available at https://console.groq.com)\n", style="muted")
    intro.append("  [2] OpenRouter ", style="bold #e4e4e7")
    intro.append("(200+ models from OpenAI, Anthropic, Meta at https://openrouter.ai)\n", style="muted")
    intro.append("  [3] Both Groq & OpenRouter ", style="bold #f4f4f5")
    intro.append("(Configure both providers for maximum model selection)\n", style="muted")
    intro.append("  [4] Skip for now ", style="dim")
    intro.append("(You can configure keys later via /key or .env)\n", style="muted")

    panel = Panel(
        intro,
        title="[brand]parallax setup[/brand] [muted]· first-run configuration[/muted]",
        title_align="left",
        box=PANEL_BOX,
        border_style="#3f3f46",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()

    choice = console.input("[brand]› Select provider [1/2/3/4] (default: 1):[/brand] ").strip() or "1"

    configured_any = False

    if choice == "1":
        key = console.input("[brand]› Enter your Groq API Key (starts with gsk_):[/brand] ").strip()
        if key:
            save_global_config({"GROQ_API_KEY": key})
            configured_any = True
            console.print(
                f"\n  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]Groq key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]."
            )
            # Offer to configure OpenRouter as well
            also = console.input("[brand]› Would you also like to configure OpenRouter? [y/N] (default: n):[/brand] ").strip().lower()
            if also in ("y", "yes"):
                or_key = console.input("[brand]› Enter your OpenRouter API Key (starts with sk-or-):[/brand] ").strip()
                if or_key:
                    save_global_config({"OPENROUTER_API_KEY": or_key})
                    console.print(
                        f"  [status.success]{ICON_SUCCESS}[/status.success] "
                        f"[muted]OpenRouter key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]."
                    )
            console.print(
                f"  [muted]Defaulting to[/muted] [tool.name]llama-3.3-70b-versatile[/tool.name] [muted](switch anytime with /model).[/muted]\n"
            )
            return True
        else:
            console.print(f"\n  [status.warning]{ICON_WARNING}[/status.warning] [muted]No key entered. Continuing without credentials.[/muted]\n")
            return False

    elif choice == "2":
        key = console.input("[brand]› Enter your OpenRouter API Key (starts with sk-or-):[/brand] ").strip()
        if key:
            save_global_config({"OPENROUTER_API_KEY": key})
            configured_any = True
            console.print(
                f"\n  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]OpenRouter key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]."
            )
            # Offer to configure Groq as well
            also = console.input("[brand]› Would you also like to configure Groq? [y/N] (default: n):[/brand] ").strip().lower()
            if also in ("y", "yes"):
                g_key = console.input("[brand]› Enter your Groq API Key (starts with gsk_):[/brand] ").strip()
                if g_key:
                    save_global_config({"GROQ_API_KEY": g_key})
                    console.print(
                        f"  [status.success]{ICON_SUCCESS}[/status.success] "
                        f"[muted]Groq key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]."
                    )
            console.print(
                f"  [muted]Defaulting to[/muted] [tool.name]liquid/lfm-2.5-2.6b:free[/tool.name] [muted](switch anytime with /model).[/muted]\n"
            )
            return True
        else:
            console.print(f"\n  [status.warning]{ICON_WARNING}[/status.warning] [muted]No key entered. Continuing without credentials.[/muted]\n")
            return False

    elif choice == "3":
        # Configure both
        g_key = console.input("[brand]› Enter your Groq API Key (starts with gsk_):[/brand] ").strip()
        if g_key:
            save_global_config({"GROQ_API_KEY": g_key})
            configured_any = True
            console.print(
                f"  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]Groq key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]."
            )

        or_key = console.input("[brand]› Enter your OpenRouter API Key (starts with sk-or-):[/brand] ").strip()
        if or_key:
            save_global_config({"OPENROUTER_API_KEY": or_key})
            configured_any = True
            console.print(
                f"  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]OpenRouter key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name]."
            )

        if configured_any:
            console.print(
                f"\n  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]Both providers configured! Defaulting to[/muted] [tool.name]llama-3.3-70b-versatile[/tool.name] "
                f"[muted](switch models anytime with /model).[/muted]\n"
            )
            return True
        else:
            console.print(f"\n  [status.warning]{ICON_WARNING}[/status.warning] [muted]No keys entered. Continuing without credentials.[/muted]\n")
            return False

    else:
        console.print(f"\n  [muted]Setup skipped. You can configure keys anytime using [tool.name]/key[/tool.name] in chat, [tool.name]parallax config[/tool.name], or a [tool.name].env[/tool.name] file.[/muted]\n")
        return False
