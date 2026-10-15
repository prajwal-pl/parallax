from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from parallax.cli.console import get_console, print_hero
from parallax.cli.doctor import run_doctor
from parallax.cli.renderer import CliRenderer
from parallax.cli.config import get_provider_status, save_global_config
from parallax.cli.theme import (
    ICON_BULLET,
    ICON_ERROR,
    ICON_SUCCESS,
    ICON_USER,
    ICON_WARNING,
    PANEL_BOX,
)
from parallax.core.context.compactor import compact
from parallax.core.llm import fetch_groq_models, fetch_openrouter_models
from parallax.core.session import Session, SessionStore
from parallax.core.types import EventType, Message

# Fallback models when offline or if live fetch fails
FALLBACK_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
FALLBACK_OPENROUTER_MODELS = [
    "liquid/lfm-2.5-2.6b:free",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
]

GROQ_MODELS: list[str] = list(FALLBACK_GROQ_MODELS)
OPENROUTER_MODELS: list[str] = list(FALLBACK_OPENROUTER_MODELS)

_MODEL_CACHE: dict[str, list[str]] = {}


@dataclass(frozen=True)
class ModelOption:
    id: str
    provider: str


async def get_provider_models(provider_name: str, force_refresh: bool = False) -> list[str]:
    """
    Dynamically fetch available models for the specified provider with in-memory caching.
    Falls back to curated list if network fails or API keys are missing.
    """
    global GROQ_MODELS, OPENROUTER_MODELS

    if not force_refresh and provider_name in _MODEL_CACHE and _MODEL_CACHE[provider_name]:
        return _MODEL_CACHE[provider_name]

    if provider_name == "GroqClient":
        try:
            live = await fetch_groq_models()
            if live:
                seen = set()
                merged = []
                for m in FALLBACK_GROQ_MODELS + live:
                    if m not in seen:
                        seen.add(m)
                        merged.append(m)
                _MODEL_CACHE[provider_name] = merged
                GROQ_MODELS = merged
                return merged
        except Exception:
            pass
        return FALLBACK_GROQ_MODELS
    else:
        try:
            live = await fetch_openrouter_models()
            if live:
                seen = set()
                merged = []
                for m in FALLBACK_OPENROUTER_MODELS + live:
                    if m not in seen:
                        seen.add(m)
                        merged.append(m)
                _MODEL_CACHE[provider_name] = merged
                OPENROUTER_MODELS = merged
                return merged
        except Exception:
            pass
        return FALLBACK_OPENROUTER_MODELS


async def get_all_models(force_refresh: bool = False) -> list[ModelOption]:
    """Fetch and merge models from both Groq and OpenRouter providers."""
    groq_models, or_models = await asyncio.gather(
        get_provider_models("GroqClient", force_refresh=force_refresh),
        get_provider_models("OpenRouterClient", force_refresh=force_refresh),
    )
    options: list[ModelOption] = []
    for m in groq_models:
        options.append(ModelOption(id=m, provider="Groq"))
    for m in or_models:
        options.append(ModelOption(id=m, provider="OpenRouter"))
    return options

if TYPE_CHECKING:
    from parallax.core.agent import Agent


class SlashCommandCompleter(Completer):
    """Interactive drop-down autocompletion menu for slash commands."""

    COMMANDS = [
        ("/help", "Show interactive guide and command help"),
        ("/help commands", "View all slash command references"),
        ("/help shortcuts", "View all keyboard shortcuts"),
        ("/model", "View or switch active LLM model"),
        ("/model groq", "Browse and select Groq models"),
        ("/model openrouter", "Browse and select OpenRouter models"),
        ("/sessions", "View all saved conversation sessions"),
        ("/session", "Switch, view, or resume sessions"),
        ("/resume", "Resume a previous session by ID"),
        ("/new", "Start a fresh new conversation session"),
        ("/key", "View or configure LLM provider API keys"),
        ("/key groq", "Set Groq API key"),
        ("/key openrouter", "Set OpenRouter API key"),
        ("/config", "Alias for /key"),
        ("/context", "Inspect conversation turns and token count"),
        ("/compact", "Manually compact conversation history"),
        ("/tools", "List all registered runtime tools"),
        ("/doctor", "Run environment and health diagnostics"),
        ("/clear", "Clear terminal screen and reset history"),
        ("/exit", "Exit the interactive session"),
        ("/quit", "Exit the interactive session"),
    ]

    def __init__(
        self,
        get_agent: Callable[[], Agent] | None = None,
        get_session_store: Callable[[], SessionStore] | None = None,
    ):
        self.get_agent = get_agent
        self.get_session_store = get_session_store

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor

        # 1. Model autocomplete after "/model " or "/m "
        if text.startswith(("/model ", "/m ")):
            prefix = "/model " if text.startswith("/model ") else "/m "
            model_text = text[len(prefix):]

            for prov_cmd in ("groq", "openrouter"):
                if prov_cmd.startswith(model_text.lower()):
                    yield Completion(
                        prov_cmd,
                        start_position=-len(model_text),
                        display=prov_cmd,
                        display_meta="Filter provider",
                    )

            all_models = (
                [(m, "Groq") for m in GROQ_MODELS]
                + [(m, "OpenRouter") for m in OPENROUTER_MODELS]
            )
            for m, provider in all_models:
                if m.lower().startswith(model_text.lower()) or model_text.lower() in m.lower():
                    yield Completion(
                        m,
                        start_position=-len(model_text),
                        display=m,
                        display_meta=provider,
                    )

        # 2. Key autocomplete after "/key ", "/keys ", or "/config "
        elif text.startswith(("/key ", "/keys ", "/config ")):
            prefix = text.split()[0] + " "
            key_text = text[len(prefix):]
            for prov in ("groq", "openrouter"):
                if prov.startswith(key_text.lower()):
                    yield Completion(
                        prov,
                        start_position=-len(key_text),
                        display=prov,
                        display_meta="Configure provider",
                    )

        # 3. Session autocomplete after "/session " or "/resume "
        elif text.startswith(("/session ", "/resume ")):
            prefix = text.split()[0] + " "
            sess_text = text[len(prefix):]
            if self.get_session_store:
                try:
                    store = self.get_session_store()
                    for s in store.list_sessions(limit=20):
                        if s.id.lower().startswith(sess_text.lower()) or sess_text.lower() in s.id.lower():
                            yield Completion(
                                s.id,
                                start_position=-len(sess_text),
                                display=s.id,
                                display_meta=f"{s.title[:24]} ({s.turn_count} turns)",
                            )
                except Exception:
                    pass

        # 4. Slash command completions
        elif text.startswith("/"):
            for cmd, desc in self.COMMANDS:
                if cmd.lower().startswith(text.lower()):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc,
                    )


# Parallax theme styling for prompt_toolkit completion popup
_PT_STYLE = Style.from_dict(
    {
        "prompt": "#e4e4e7 bold",
        "completion-menu": "bg:#18181b #a1a1aa",
        "completion-menu.completion": "bg:#27272a #f4f4f5",
        "completion-menu.completion.current": "bg:#3f3f46 #ffffff bold",
        "completion-menu.meta": "bg:#18181b #71717a",
        "completion-menu.meta.current": "bg:#3f3f46 #d4d4d8 italic",
        "scrollbar.background": "bg:#18181b",
        "scrollbar.button": "bg:#3f3f46",
    }
)


class InteractiveRepl:
    """
    Interactive terminal REPL providing a multi-turn conversation loop,
    slash commands with autocomplete popup, and live event rendering.
    """

    def __init__(
        self,
        agent: Agent,
        console: Console | None = None,
        verbose: bool = False,
        session: Session | None = None,
        session_store: SessionStore | None = None,
    ):
        self.agent = agent
        self.console = console or get_console()
        self.verbose = verbose
        self.session_store = session_store or SessionStore()

        if session is not None:
            self.session = session
            # Load non-system messages into REPL turn history
            self.messages: list[Message] = [
                m for m in self.session.messages if m.role != "system"
            ]
        else:
            self.session = self.session_store.create_session(
                workspace=str(self.agent.workspace),
                model=self.agent.model,
            )
            self.messages = []

        self._prompt_session: PromptSession[str] | None = None
        self._prefetch_task: asyncio.Task[None] | None = None

    async def _prefetch_models(self) -> None:
        """Prefetch live model lists in the background on session startup."""
        try:
            await get_all_models()
        except Exception:
            pass

    def _get_prompt_session(self) -> PromptSession[str]:
        if self._prompt_session is None:
            self._prompt_session = PromptSession(
                completer=SlashCommandCompleter(
                    get_agent=lambda: self.agent,
                    get_session_store=lambda: self.session_store,
                ),
                complete_while_typing=True,
                style=_PT_STYLE,
            )
        return self._prompt_session

    async def run(self) -> None:
        """Run the interactive REPL loop."""
        print_hero(
            model=self.agent.model,
            workspace=self.agent.workspace,
            session_id=self.session.id,
            console=self.console,
        )

        if self.messages:
            self.console.print(
                f"  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]resumed session[/muted] [tool.name]{self.session.id}[/tool.name] "
                f"[dim]({len(self.messages)} messages loaded · {self.session.title})[/dim]\n"
            )
            renderer = CliRenderer(console=self.console, verbose=self.verbose, is_interactive=True)
            renderer.render_history(self.messages)

        # Warm the dynamic model cache in the background
        self._prefetch_task = asyncio.create_task(self._prefetch_models())

        while True:
            try:
                # Use prompt_toolkit in interactive terminal, fallback to console.input if non-tty pipe
                if sys.stdin.isatty():
                    session = self._get_prompt_session()
                    user_input = (await session.prompt_async(f"{ICON_USER} ")).strip()
                else:
                    user_input = self.console.input(f"[user]{ICON_USER}[/user] ").strip()
            except (KeyboardInterrupt, EOFError, OSError):
                self.console.print("\n[dim #64748b]session ended[/dim #64748b]")
                break
            except Exception:
                # Fallback to rich input if prompt_toolkit encounters any issue
                try:
                    user_input = self.console.input(f"[user]{ICON_USER}[/user] ").strip()
                except (KeyboardInterrupt, EOFError, OSError):
                    self.console.print("\n[dim #64748b]session ended[/dim #64748b]")
                    break

            if not user_input:
                continue

            # Check for slash commands
            if user_input.startswith("/"):
                handled = await self._handle_command(user_input)
                if handled == "exit":
                    self.console.print("[dim #64748b]session ended[/dim #64748b]")
                    break
                continue

            # Execute conversational turn
            await self._run_turn(user_input)

    async def _run_turn(self, user_input: str) -> None:
        """Run a single agent turn with live event streaming and history persistence."""
        renderer = CliRenderer(console=self.console, verbose=self.verbose, is_interactive=True)
        stream = self.agent.run_stream(user_input, history=self.messages)

        try:
            async for event in stream:
                renderer.handle_event(event)
                # Capture updated conversation history on completion
                if event.type == EventType.RUN_COMPLETED:
                    updated_messages = event.data.get("messages")
                    if updated_messages:
                        self.messages = updated_messages
                        # Persist session state
                        self.session.update(self.messages, model=self.agent.model)
                        self.session_store.save_session(self.session)
        except (KeyboardInterrupt, asyncio.CancelledError):
            renderer.stop_spinner()
            self.console.print(f"\n  [status.warning]! run cancelled[/status.warning]\n")
        except Exception as e:
            renderer.stop_spinner()
            renderer._render_abstracted_error(str(e), duration=0.0)

    async def _handle_command(self, cmd_line: str) -> str | None:
        """Handle slash command execution with aliases."""
        parts = cmd_line.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command in ("/exit", "/quit", "/q"):
            return "exit"

        elif command in ("/help", "/h", "?"):
            self._show_help(topic=args)

        elif command in ("/clear", "/cls"):
            if self.messages:
                self.session.update(self.messages, model=self.agent.model)
                self.session_store.save_session(self.session)
            self.session = self.session_store.create_session(
                workspace=str(self.agent.workspace),
                model=self.agent.model,
            )
            self.messages = []
            self.console.clear()
            print_hero(
                model=self.agent.model,
                workspace=self.agent.workspace,
                session_id=self.session.id,
                console=self.console,
            )

        elif command == "/sessions":
            self._show_sessions()

        elif command in ("/resume", "/session"):
            if not args or args.lower() in ("list", "ls"):
                self._show_sessions()
            elif args.lower() in ("new", "create"):
                self._new_session()
            else:
                self._resume_session(args)

        elif command == "/new":
            self._new_session()

        elif command in ("/model", "/m"):
            norm_args = args.lower().strip()
            if norm_args in ("groq", "g"):
                old_model = self.agent.model
                if sys.stdin.isatty() and (not _MODEL_CACHE.get("GroqClient") or not _MODEL_CACHE.get("OpenRouterClient")):
                    with self.console.status("[dim]loading models...[/dim]", spinner="dots"):
                        models = await get_all_models()
                else:
                    models = await get_all_models()
                selected = await self._pick_model(models, initial_provider_filter="Groq")
                await self._apply_selected_model(selected, old_model)

            elif norm_args in ("openrouter", "or"):
                old_model = self.agent.model
                if sys.stdin.isatty() and (not _MODEL_CACHE.get("GroqClient") or not _MODEL_CACHE.get("OpenRouterClient")):
                    with self.console.status("[dim]loading models...[/dim]", spinner="dots"):
                        models = await get_all_models()
                else:
                    models = await get_all_models()
                selected = await self._pick_model(models, initial_provider_filter="OpenRouter")
                await self._apply_selected_model(selected, old_model)

            elif args:
                old_model = self.agent.model
                self.agent.set_model(args)
                if self.agent.model != old_model:
                    self.session.model = self.agent.model
                    self.session_store.save_session(self.session)
                self.console.print(
                    f"[status.success]{ICON_SUCCESS}[/status.success] "
                    f"[muted]switched model from[/muted] [tool.name]{old_model}[/tool.name] "
                    f"[muted]to[/muted] [tool.name]{self.agent.model}[/tool.name] "
                    f"[dim]({self.agent.provider_name})[/dim]\n"
                )
                self._warn_if_key_missing()

            else:
                old_model = self.agent.model
                if sys.stdin.isatty() and (not _MODEL_CACHE.get("GroqClient") or not _MODEL_CACHE.get("OpenRouterClient")):
                    with self.console.status("[dim]loading models...[/dim]", spinner="dots"):
                        models = await get_all_models()
                else:
                    models = await get_all_models()
                selected = await self._pick_model(models, initial_provider_filter="All")
                await self._apply_selected_model(selected, old_model)

        elif command in ("/key", "/keys", "/config"):
            await self._handle_key_command(args)

        elif command in ("/doctor", "/d"):
            run_doctor(self.console)

        elif command in ("/context", "/c"):
            self._show_context()

        elif command == "/compact":
            self._compact_context()

        elif command in ("/tools", "/t"):
            self._show_tools()

        else:
            self.console.print(
                f"[status.error]{ICON_ERROR}[/status.error] [muted]unknown command '{command}'. Type /help for available commands.[/muted]\n"
            )

        return None

    def _warn_if_key_missing(self) -> None:
        """Warn user if active model's provider has no configured API key."""
        import os
        if self.agent.provider_name == "Groq" and not os.getenv("GROQ_API_KEY"):
            self.console.print(
                f"  [status.warning]{ICON_WARNING}[/status.warning] "
                f"[muted]Warning: GROQ_API_KEY is not configured. Run[/muted] "
                f"[tool.name]/key groq <api-key>[/tool.name] [muted]to add it.[/muted]\n"
            )
        elif self.agent.provider_name == "OpenRouter" and not os.getenv("OPENROUTER_API_KEY"):
            self.console.print(
                f"  [status.warning]{ICON_WARNING}[/status.warning] "
                f"[muted]Warning: OPENROUTER_API_KEY is not configured. Run[/muted] "
                f"[tool.name]/key openrouter <api-key>[/tool.name] [muted]to add it.[/muted]\n"
            )

    async def _handle_key_command(self, args: str) -> None:
        """Handle /key slash command to view or configure provider keys."""
        args_clean = args.strip()
        if not args_clean:
            statuses = get_provider_status()
            table = Table(
                box=None,
                show_header=True,
                header_style="bold #e4e4e7",
                expand=True,
                padding=(0, 2),
            )
            table.add_column("Provider", style="tool.name", width=14)
            table.add_column("Status", style="status.info", width=16)
            table.add_column("Key", style="white", width=16)
            table.add_column("Description", style="muted")

            for k, p in statuses.items():
                status_txt = (
                    f"[status.success]{ICON_SUCCESS} configured[/status.success]"
                    if p["configured"]
                    else f"[status.warning]{ICON_WARNING} unset[/status.warning]"
                )
                table.add_row(
                    p["name"],
                    status_txt,
                    p["masked"],
                    p["description"],
                )

            panel = Panel(
                table,
                box=PANEL_BOX,
                border_style="#334155",
                padding=(0, 1),
                title="[brand]parallax[/brand] [dim #64748b]· provider api keys[/dim #64748b]",
                title_align="left",
            )
            self.console.print()
            self.console.print(panel)
            self.console.print("[dim #64748b]To set a key: /key groq <key> or /key openrouter <key>[/dim #64748b]\n")

            if sys.stdin.isatty():
                try:
                    choice = self.console.input("[brand]› Configure a key now? [1: Groq, 2: OpenRouter, Enter to skip]:[/brand] ").strip()
                    if choice == "1":
                        k = self.console.input("[brand]› Enter Groq API Key (starts with gsk_):[/brand] ").strip()
                        if k:
                            save_global_config({"GROQ_API_KEY": k})
                            _MODEL_CACHE.clear()
                            self.console.print(f"  [status.success]{ICON_SUCCESS}[/status.success] [muted]Groq key saved and activated.[/muted]\n")
                    elif choice == "2":
                        k = self.console.input("[brand]› Enter OpenRouter API Key (starts with sk-or-):[/brand] ").strip()
                        if k:
                            save_global_config({"OPENROUTER_API_KEY": k})
                            _MODEL_CACHE.clear()
                            self.console.print(f"  [status.success]{ICON_SUCCESS}[/status.success] [muted]OpenRouter key saved and activated.[/muted]\n")
                except (KeyboardInterrupt, EOFError):
                    self.console.print()
            return

        parts = args_clean.split(maxsplit=1)
        provider = parts[0].lower()
        if len(parts) > 1:
            key_val = parts[1].strip()
        else:
            if sys.stdin.isatty():
                key_val = self.console.input(f"[brand]› Enter {provider.capitalize()} API Key:[/brand] ").strip()
            else:
                self.console.print(f"[status.error]{ICON_ERROR}[/status.error] [muted]Usage: /key {provider} <api-key>[/muted]\n")
                return

        if not key_val:
            self.console.print(f"[status.warning]{ICON_WARNING}[/status.warning] [muted]No key provided.[/muted]\n")
            return

        if provider in ("groq", "g"):
            save_global_config({"GROQ_API_KEY": key_val})
            _MODEL_CACHE.clear()
            self.console.print(
                f"[status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]Groq API key saved to ~/.parallax/config.json and activated.[/muted]\n"
            )
        elif provider in ("openrouter", "or"):
            save_global_config({"OPENROUTER_API_KEY": key_val})
            _MODEL_CACHE.clear()
            self.console.print(
                f"[status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]OpenRouter API key saved to ~/.parallax/config.json and activated.[/muted]\n"
            )
        else:
            self.console.print(
                f"[status.error]{ICON_ERROR}[/status.error] "
                f"[muted]Unknown provider '{provider}'. Use 'groq' or 'openrouter'.[/muted]\n"
            )

    def _show_sessions(self) -> None:
        """Render a table of saved conversation sessions."""
        sessions = self.session_store.list_sessions(limit=25)
        if not sessions:
            self.console.print("[dim #64748b]No saved sessions found.[/dim #64748b]\n")
            return

        table = Table(
            box=None,
            show_header=True,
            header_style="bold #e4e4e7",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Session ID", style="brand", no_wrap=True)
        table.add_column("Title", style="white", ratio=2)
        table.add_column("Model", style="tool.name", ratio=1)
        table.add_column("Turns", style="dim #71717a", width=6, justify="right")
        table.add_column("Updated", style="dim #71717a", no_wrap=True)

        for s in sessions:
            is_active = (s.id == self.session.id)
            id_text = f"[bold #22c55e]●[/bold #22c55e] [bold white]{s.id}[/bold white]" if is_active else f"  {s.id}"
            title_text = s.title if len(s.title) <= 36 else s.title[:33] + "..."
            updated_text = s.updated_at[:16].replace("T", " ")
            table.add_row(
                id_text,
                title_text,
                s.model or "default",
                str(s.turn_count),
                updated_text,
            )

        panel = Panel(
            table,
            box=PANEL_BOX,
            border_style="#3f3f46",
            padding=(0, 1),
            title="[bold #e4e4e7]parallax[/bold #e4e4e7] [dim #71717a]· conversation sessions[/dim #71717a]",
            title_align="left",
        )
        self.console.print()
        self.console.print(panel)
        self.console.print(
            "[dim #71717a]● active · Switch/resume: /resume <id> (or prefix) · Fresh: /new[/dim #71717a]\n"
        )

    def _resume_session(self, target_id: str) -> None:
        """Switch active session to a previously saved session."""
        target_id = target_id.strip()
        if not target_id:
            self._show_sessions()
            if sys.stdin.isatty():
                try:
                    choice = self.console.input("[brand]› Enter Session ID or prefix to resume:[/brand] ").strip()
                    if choice:
                        self._resume_session(choice)
                except (KeyboardInterrupt, EOFError):
                    self.console.print()
            return

        loaded = self.session_store.get_session(target_id)
        if not loaded:
            self.console.print(
                f"[status.error]{ICON_ERROR}[/status.error] "
                f"[muted]session matching '{target_id}' not found. Type /sessions to view all.[/muted]\n"
            )
            return

        if loaded.id == self.session.id:
            self.console.print(
                f"[dim #64748b]Session {loaded.id} is already active.[/dim #64748b]\n"
            )
            return

        # Auto-save current session before switching
        if self.messages:
            self.session.update(self.messages, model=self.agent.model)
            self.session_store.save_session(self.session)

        # Switch to loaded session
        self.session = loaded
        self.messages = [m for m in self.session.messages if m.role != "system"]
        if self.session.model:
            try:
                self.agent.set_model(self.session.model)
            except Exception:
                pass

        # Erase existing session messages visually and reset terminal buffer
        self.console.clear()
        print_hero(
            model=self.agent.model,
            workspace=self.agent.workspace,
            session_id=self.session.id,
            console=self.console,
        )

        self.console.print(
            f"  [status.success]{ICON_SUCCESS}[/status.success] "
            f"[muted]resumed session[/muted] [tool.name]{self.session.id}[/tool.name] "
            f"[dim]({len(self.messages)} messages · {self.session.turn_count} turns · model: {self.session.model})[/dim]"
        )
        if self.session.title:
            self.console.print(f"  [muted]Title:[/muted] [white]{self.session.title}[/white]")
        self.console.print()

        # Render all past messages from the resumed session
        renderer = CliRenderer(console=self.console, verbose=self.verbose, is_interactive=True)
        renderer.render_history(self.messages)

    def _new_session(self) -> None:
        """Save current session and start a new clean session."""
        if self.messages:
            self.session.update(self.messages, model=self.agent.model)
            self.session_store.save_session(self.session)

        self.session = self.session_store.create_session(
            workspace=str(self.agent.workspace),
            model=self.agent.model,
        )
        self.messages = []

        self.console.clear()
        print_hero(
            model=self.agent.model,
            workspace=self.agent.workspace,
            session_id=self.session.id,
            console=self.console,
        )
        self.console.print(
            f"  [status.success]{ICON_SUCCESS}[/status.success] "
            f"[muted]started new session[/muted] [tool.name]{self.session.id}[/tool.name]\n"
        )

    def _create_commands_table(self) -> Table:
        table = Table(
            box=None,
            show_header=True,
            header_style="bold #e4e4e7",
            expand=True,
            padding=(0, 2),
        )
        table.add_column("Command", style="brand", width=18)
        table.add_column("Description", style="white")
        table.add_column("Aliases", style="dim #71717a", width=12)

        table.add_row("/help [topic]", "Show interactive guide and command help", "/h, ?")
        table.add_row("/model [name]", "View or switch active LLM provider model", "/m")
        table.add_row("/sessions", "View all saved conversation sessions", "")
        table.add_row("/resume <id>", "Resume a previous session by ID or prefix", "/session")
        table.add_row("/new", "Start a fresh new conversation session", "")
        table.add_row("/key [provider]", "View or configure LLM provider API keys", "/keys, /config")
        table.add_row("/context", "Inspect conversation turns and token usage", "/c")
        table.add_row("/compact", "Trigger turn-atomic history compaction", "")
        table.add_row("/tools", "List registered runtime tools and schemas", "/t")
        table.add_row("/doctor", "Run environment, API key, and Git diagnostics", "/d")
        table.add_row("/clear", "Clear screen and start a fresh session", "/cls")
        table.add_row("/exit", "Exit the interactive session", "/quit, /q")
        return table

    def _create_shortcuts_table(self) -> Table:
        table = Table(
            box=None,
            show_header=True,
            header_style="bold #e4e4e7",
            expand=True,
            padding=(0, 2),
        )
        table.add_column("Shortcut", style="brand", width=18)
        table.add_column("Action", style="white")
        table.add_column("Context", style="dim #71717a", width=14)

        table.add_row("/", "Trigger interactive command suggestions", "Input prompt")
        table.add_row("Up / Down", "Navigate command suggestions or history", "Input prompt")
        table.add_row("Tab / Enter", "Accept selected command suggestion", "Autocomplete")
        table.add_row("Ctrl + C", "Interrupt current task without exiting", "Execution")
        table.add_row("Ctrl + D", "Exit interactive session cleanly", "Input prompt")
        return table

    def _show_help(self, topic: str = "") -> None:
        """Render clean, accessible help reference with topic filtering."""
        norm_topic = topic.strip().lower()

        if norm_topic in ("commands", "command", "cmd", "cmds", "c"):
            panel = Panel(
                self._create_commands_table(),
                box=PANEL_BOX,
                border_style="#3f3f46",
                padding=(0, 1),
                title="[brand]parallax[/brand] [dim #71717a]· command reference[/dim #71717a]",
                title_align="left",
            )
            self.console.print()
            self.console.print(panel)
            self.console.print("[dim #71717a]Type /help shortcuts for keyboard bindings or /help for full guide[/dim #71717a]\n")

        elif norm_topic in ("shortcuts", "shortcut", "keys", "key", "s"):
            panel = Panel(
                self._create_shortcuts_table(),
                box=PANEL_BOX,
                border_style="#3f3f46",
                padding=(0, 1),
                title="[brand]parallax[/brand] [dim #71717a]· keyboard shortcuts[/dim #71717a]",
                title_align="left",
            )
            self.console.print()
            self.console.print(panel)
            self.console.print("[dim #71717a]Type /help commands for slash commands or /help for full guide[/dim #71717a]\n")

        elif not norm_topic or norm_topic in ("all", "guide"):
            grid = Table.grid(padding=(1, 0))
            grid.add_column()
            grid.add_row(Text("Commands", style="bold #f4f4f5"))
            grid.add_row(self._create_commands_table())
            grid.add_row(Text("Keyboard Shortcuts", style="bold #f4f4f5"))
            grid.add_row(self._create_shortcuts_table())
            grid.add_row(Text("Tip: Filter help with /help commands or /help shortcuts", style="dim #71717a"))

            panel = Panel(
                grid,
                box=PANEL_BOX,
                border_style="#3f3f46",
                padding=(0, 1),
                title="[brand]parallax guide[/brand]",
                title_align="left",
            )
            self.console.print()
            self.console.print(panel)
            self.console.print()

        else:
            self.console.print(
                f"[status.error]{ICON_ERROR}[/status.error] [muted]unknown help topic '{topic}'. "
                f"Available topics: [brand]commands[/brand], [brand]shortcuts[/brand] (or run [brand]/help[/brand] for full guide).[/muted]\n"
            )

    def _show_context(self) -> None:
        turns = len(self.messages)
        token_count = self.agent.context.current_tokens(self.messages) if self.messages else 0
        limit = self.agent.context.config.max_context_tokens

        msg = Text()
        msg.append(f"{ICON_BULLET} ", style="brand")
        msg.append("active messages: ", style="muted")
        msg.append(str(turns), style="tool.name")
        msg.append(f"  {ICON_BULLET}  ", style="muted")
        msg.append("estimated tokens: ", style="muted")
        msg.append(f"{token_count:,} / {limit:,}", style="tool.name")
        self.console.print(msg)
        self.console.print()

    def _compact_context(self) -> None:
        if not self.messages:
            self.console.print("[muted]no history to compact[/muted]\n")
            return

        cfg = self.agent.context.config
        before = self.agent.context.current_tokens(self.messages)
        compacted, summarized = compact(
            self.messages, cfg.keep_recent_turns, cfg.summary_cap_chars
        )
        if summarized > 0:
            self.messages = compacted
            after = self.agent.context.current_tokens(self.messages)
            self.console.print(
                f"[status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]compacted history: {before:,} {ICON_BULLET} {after:,} tokens ({summarized} turns summarized)[/muted]\n"
            )
        else:
            self.console.print("[muted]history already compact[/muted]\n")

    def _show_tools(self) -> None:
        """Render active tools in styled cards with parameter highlights."""
        self.console.print()
        self.console.print(
            f"[bold #e4e4e7]Active Runtime Tools[/bold #e4e4e7] "
            f"[dim #71717a]({len(self.agent.tool_schemas)} tools available)[/dim #71717a]\n"
        )

        for schema in self.agent.tool_schemas:
            func = schema.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "").splitlines()[0] if func.get("description") else ""
            params = func.get("parameters", {}).get("properties", {})

            content = Text()
            content.append(f"{desc}\n\n", style="muted")
            content.append("Parameters: ", style="bold #a1a1aa")
            if params:
                for param_name in params.keys():
                    content.append(f" {param_name} ", style="bold #f4f4f5 on #27272a")
                    content.append(" ")
            else:
                content.append("none", style="dim")
            content.append("\n")

            panel = Panel(
                content,
                title=f"[bold #e4e4e7]{name}[/bold #e4e4e7]",
                title_align="left",
                box=PANEL_BOX,
                border_style="#3f3f46",
                padding=(0, 2),
            )
            self.console.print(panel)
        self.console.print()

    async def _apply_selected_model(
        self, selected: ModelOption | str | None, old_model: str
    ) -> None:
        """Apply selected model and display confirmation."""
        if isinstance(selected, ModelOption):
            self.agent.set_model(selected.id, provider=selected.provider)
            if self.agent.model != old_model:
                self.session.model = self.agent.model
                self.session_store.save_session(self.session)
            self.console.print(
                f"[status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]switched model from[/muted] [tool.name]{old_model}[/tool.name] "
                f"[muted]to[/muted] [tool.name]{self.agent.model}[/tool.name] "
                f"[dim]({self.agent.provider_name})[/dim]\n"
            )
            self._warn_if_key_missing()
        elif isinstance(selected, str):
            if selected == "__custom__":
                session = self._get_prompt_session()
                custom_name = await session.prompt_async("Enter custom model name: ")
                selected = custom_name.strip() if custom_name and custom_name.strip() else None
            if selected:
                self.agent.set_model(selected)
                if self.agent.model != old_model:
                    self.session.model = self.agent.model
                    self.session_store.save_session(self.session)
                self.console.print(
                    f"[status.success]{ICON_SUCCESS}[/status.success] "
                    f"[muted]switched model from[/muted] [tool.name]{old_model}[/tool.name] "
                    f"[muted]to[/muted] [tool.name]{self.agent.model}[/tool.name] "
                    f"[dim]({self.agent.provider_name})[/dim]\n"
                )
                self._warn_if_key_missing()
            elif sys.stdin.isatty():
                self.console.print(
                    f"[dim #64748b]model selection cancelled (current: {self.agent.model})[/dim #64748b]\n"
                )
        elif sys.stdin.isatty():
            self.console.print(
                f"[dim #64748b]model selection cancelled (current: {self.agent.model})[/dim #64748b]\n"
            )

    async def _pick_model(
        self,
        models: list[ModelOption] | list[str],
        initial_provider_filter: str = "All",
    ) -> ModelOption | str | None:
        """
        Interactive inline model picker that blocks until the user selects a model
        with arrow keys and Enter, or cancels with Esc / Ctrl+C.
        Supports provider tabs (Tab key: All / Groq / OpenRouter) and real-time search filtering.
        """
        # Normalize to list of ModelOption
        model_options: list[ModelOption] = []
        for m in models:
            if isinstance(m, ModelOption):
                model_options.append(m)
            else:
                prov = "Groq" if (m.startswith("groq/") or m in FALLBACK_GROQ_MODELS or m in GROQ_MODELS) else "OpenRouter"
                model_options.append(ModelOption(id=m, provider=prov))

        # Prioritize current model at top of list
        active_opt: ModelOption | None = None
        other_opts: list[ModelOption] = []
        for opt in model_options:
            if opt.id == self.agent.model and active_opt is None:
                active_opt = opt
            else:
                other_opts.append(opt)
        sorted_options = ([active_opt] if active_opt else []) + other_opts

        if not sys.stdin.isatty():
            self.console.print(
                f"[muted]current model:[/muted] [tool.name]{self.agent.model}[/tool.name] "
                f"[dim]({self.agent.provider_name})[/dim]\n"
            )
            groq_sample = [m.id for m in sorted_options if m.provider == "Groq"][:5]
            or_sample = [m.id for m in sorted_options if m.provider == "OpenRouter"][:5]
            self.console.print("[bold #e4e4e7]Groq models:[/bold #e4e4e7]")
            for m in groq_sample:
                marker = f"  {ICON_USER} " if m == self.agent.model else "    "
                self.console.print(f"{marker}[tool.name]{m}[/tool.name]")
            self.console.print("\n[bold #e4e4e7]OpenRouter models:[/bold #e4e4e7]")
            for m in or_sample:
                marker = f"  {ICON_USER} " if m == self.agent.model else "    "
                self.console.print(f"{marker}[tool.name]{m}[/tool.name]")
            self.console.print("\n[dim]To switch model directly, run: /model <name>[/dim]\n")
            return None

        tabs = ["All", "Groq", "OpenRouter"]
        provider_filter = initial_provider_filter if initial_provider_filter in tabs else "All"
        query = ""
        selected_idx = 0
        PAGE_SIZE = 8

        def get_filtered() -> list[ModelOption]:
            if provider_filter == "All":
                candidates = sorted_options
            else:
                candidates = [m for m in sorted_options if m.provider == provider_filter]

            if not query:
                return candidates
            q = query.lower()
            return [
                m for m in candidates
                if q in m.id.lower() or q in m.provider.lower()
            ]

        kb = KeyBindings()

        @kb.add("tab")
        def _next_tab(event):
            nonlocal provider_filter, selected_idx
            idx = tabs.index(provider_filter)
            provider_filter = tabs[(idx + 1) % len(tabs)]
            selected_idx = 0

        @kb.add("s-tab")
        def _prev_tab(event):
            nonlocal provider_filter, selected_idx
            idx = tabs.index(provider_filter)
            provider_filter = tabs[(idx - 1) % len(tabs)]
            selected_idx = 0

        @kb.add("up")
        def _move_up(event):
            nonlocal selected_idx
            f = get_filtered()
            if f:
                selected_idx = max(0, selected_idx - 1)

        @kb.add("down")
        def _move_down(event):
            nonlocal selected_idx
            f = get_filtered()
            if f:
                selected_idx = min(len(f) - 1, selected_idx + 1)

        @kb.add("pageup")
        def _page_up(event):
            nonlocal selected_idx
            f = get_filtered()
            if f:
                selected_idx = max(0, selected_idx - PAGE_SIZE)

        @kb.add("pagedown")
        def _page_down(event):
            nonlocal selected_idx
            f = get_filtered()
            if f:
                selected_idx = min(len(f) - 1, selected_idx + PAGE_SIZE)

        @kb.add("enter")
        def _confirm(event):
            f = get_filtered()
            if f and 0 <= selected_idx < len(f):
                event.app.exit(result=f[selected_idx])
            elif query.strip():
                event.app.exit(result=query.strip())
            else:
                event.app.exit(result=None)

        @kb.add("backspace")
        def _backspace(event):
            nonlocal query, selected_idx
            query = query[:-1]
            selected_idx = 0

        @kb.add("c-u")
        @kb.add("c-w")
        def _clear_query(event):
            nonlocal query, selected_idx
            query = ""
            selected_idx = 0

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event):
            event.app.exit(result=None)

        @kb.add("<any>")
        def _type_char(event):
            nonlocal query, selected_idx
            for key in event.key_sequence:
                if key.data and key.data.isprintable():
                    query += key.data
                    selected_idx = 0

        total_all = len(sorted_options)
        total_groq = sum(1 for m in sorted_options if m.provider == "Groq")
        total_or = sum(1 for m in sorted_options if m.provider == "OpenRouter")

        def get_formatted_text() -> StyleAndTextTuples:
            f = get_filtered()
            tokens: StyleAndTextTuples = [
                ("class:header", "  Select Model\n"),
                ("class:tab_lbl", "  Provider: "),
            ]

            for t in tabs:
                count = total_all if t == "All" else (total_groq if t == "Groq" else total_or)
                if t == provider_filter:
                    tokens.append(("class:tab_active", f"[{t} ({count})] "))
                else:
                    tokens.append(("class:tab_inactive", f" {t} ({count})  "))
            tokens.append(("class:hint_subtle", "(Tab to switch)\n"))

            tokens.append(("class:filter_lbl", "  Filter: "))
            tokens.append(("class:filter_val", f"{query}█\n\n"))

            if f:
                offset = max(0, min(selected_idx - PAGE_SIZE // 2, len(f) - PAGE_SIZE))
                visible = f[offset : offset + PAGE_SIZE]

                if offset > 0:
                    tokens.append(("class:hint_subtle", f"    ↑ {offset} more above\n"))

                for i, opt in enumerate(visible):
                    real_idx = offset + i
                    is_selected = (real_idx == selected_idx)
                    is_current = (opt.id == self.agent.model)
                    active_tag = " (active)" if is_current else ""

                    name_str = f"{opt.id}{active_tag}"
                    prov_badge = f"[{opt.provider}]"
                    spacing = max(2, 46 - len(name_str))

                    if is_selected:
                        tokens.append(("class:pointer", "  › "))
                        tokens.append(("class:selected", f"{name_str}{' ' * spacing}{prov_badge}\n"))
                    else:
                        tokens.append(("class:pointer_inactive", "    "))
                        tokens.append(("class:unselected", f"{name_str}{' ' * spacing}"))
                        tokens.append(("class:provider_tag", f"{prov_badge}\n"))

                if offset + len(visible) < len(f):
                    remaining = len(f) - (offset + len(visible))
                    tokens.append(("class:hint_subtle", f"    ↓ {remaining} more below\n"))
            else:
                tokens.append(
                    ("class:hint", f'  No models match "{query}" in {provider_filter}. Press Enter to use as custom model.\n')
                )

            tokens.append(
                ("class:hint", "\n  Type to filter · Tab switch provider · ↑/↓ navigate · Enter select · Esc cancel\n")
            )
            return tokens

        control = FormattedTextControl(get_formatted_text)
        window = Window(content=control, height=PAGE_SIZE + 7)
        layout = Layout(HSplit([window]))

        style = Style.from_dict({
            "header": "#e4e4e7 bold",
            "tab_lbl": "#a1a1aa bold",
            "tab_active": "#ffffff bold bg:#27272a",
            "tab_inactive": "#71717a",
            "filter_lbl": "#a1a1aa bold",
            "filter_val": "#f4f4f5 bold",
            "pointer": "#f4f4f5 bold",
            "pointer_inactive": "",
            "selected": "#ffffff bold bg:#27272a",
            "unselected": "#a1a1aa",
            "provider_tag": "#71717a",
            "hint_subtle": "#52525b italic",
            "hint": "#71717a italic",
        })

        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            erase_when_done=True,
        )

        try:
            choice = await app.run_async()
        except (KeyboardInterrupt, EOFError):
            return None

        return choice
