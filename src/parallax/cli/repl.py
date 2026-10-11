from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from parallax.cli.console import get_console, print_hero
from parallax.cli.doctor import run_doctor
from parallax.cli.renderer import CliRenderer
from parallax.cli.theme import (
    ICON_BULLET,
    ICON_ERROR,
    ICON_SUCCESS,
    ICON_USER,
    PANEL_BOX,
)
from parallax.core.context.compactor import compact
from parallax.core.types import EventType, Message

if TYPE_CHECKING:
    from parallax.core.agent import Agent


class SlashCommandCompleter(Completer):
    """Interactive drop-down autocompletion menu for slash commands."""

    COMMANDS = [
        ("/help", "Show interactive guide and command help"),
        ("/help commands", "View all slash command references"),
        ("/help shortcuts", "View all keyboard shortcuts"),
        ("/model", "View or switch active LLM model"),
        ("/context", "Inspect conversation turns and token count"),
        ("/compact", "Manually compact conversation history"),
        ("/tools", "List all registered runtime tools"),
        ("/doctor", "Run environment and health diagnostics"),
        ("/clear", "Clear terminal screen and reset history"),
        ("/exit", "Exit the interactive session"),
        ("/quit", "Exit the interactive session"),
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
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
        "prompt": "#38bdf8 bold",
        "completion-menu": "bg:#0f172a #94a3b8",
        "completion-menu.completion": "bg:#1e293b #e2e8f0",
        "completion-menu.completion.current": "bg:#4f46e5 #ffffff bold",
        "completion-menu.meta": "bg:#0f172a #64748b",
        "completion-menu.meta.current": "bg:#4f46e5 #c7d2fe italic",
        "scrollbar.background": "bg:#0f172a",
        "scrollbar.button": "bg:#334155",
    }
)


class InteractiveRepl:
    """
    Interactive terminal REPL providing a multi-turn conversation loop,
    slash commands with autocomplete popup, and live event rendering.
    """

    def __init__(self, agent: Agent, console: Console | None = None, verbose: bool = False):
        self.agent = agent
        self.console = console or get_console()
        self.verbose = verbose
        self.messages: list[Message] = []
        self._prompt_session: PromptSession[str] | None = None

    def _get_prompt_session(self) -> PromptSession[str]:
        if self._prompt_session is None:
            self._prompt_session = PromptSession(
                completer=SlashCommandCompleter(),
                complete_while_typing=True,
                style=_PT_STYLE,
            )
        return self._prompt_session

    async def run(self) -> None:
        """Run the interactive REPL loop."""
        print_hero(model=self.agent.model, console=self.console)

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
            self.messages.clear()
            self.console.clear()
            print_hero(model=self.agent.model, console=self.console)

        elif command in ("/model", "/m"):
            if args:
                old_model = self.agent.model
                self.agent.model = args
                self.console.print(
                    f"[status.success]{ICON_SUCCESS}[/status.success] "
                    f"[muted]switched model from[/muted] [tool.name]{old_model}[/tool.name] "
                    f"[muted]to[/muted] [tool.name]{args}[/tool.name]\n"
                )
            else:
                self.console.print(f"[muted]current model:[/muted] [tool.name]{self.agent.model}[/tool.name]\n")

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

    def _create_commands_table(self) -> Table:
        table = Table(
            box=None,
            show_header=True,
            header_style="bold #818cf8",
            expand=True,
            padding=(0, 2),
        )
        table.add_column("Command", style="brand", width=18)
        table.add_column("Description", style="white")
        table.add_column("Aliases", style="dim #94a3b8", width=12)

        table.add_row("/help [topic]", "Show interactive guide and command help", "/h, ?")
        table.add_row("/model [name]", "View or switch active LLM provider model", "/m")
        table.add_row("/context", "Inspect conversation turns and token usage", "/c")
        table.add_row("/compact", "Trigger turn-atomic history compaction", "")
        table.add_row("/tools", "List registered runtime tools and schemas", "/t")
        table.add_row("/doctor", "Run environment, API key, and Git diagnostics", "/d")
        table.add_row("/clear", "Clear screen and reset session history", "/cls")
        table.add_row("/exit", "Exit the interactive session", "/quit, /q")
        return table

    def _create_shortcuts_table(self) -> Table:
        table = Table(
            box=None,
            show_header=True,
            header_style="bold #818cf8",
            expand=True,
            padding=(0, 2),
        )
        table.add_column("Shortcut", style="brand", width=18)
        table.add_column("Action", style="white")
        table.add_column("Context", style="dim #94a3b8", width=14)

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
                border_style="#334155",
                padding=(0, 1),
                title="[brand]parallax[/brand] [dim #64748b]· command reference[/dim #64748b]",
                title_align="left",
            )
            self.console.print()
            self.console.print(panel)
            self.console.print("[dim #64748b]Type /help shortcuts for keyboard bindings or /help for full guide[/dim #64748b]\n")

        elif norm_topic in ("shortcuts", "shortcut", "keys", "key", "s"):
            panel = Panel(
                self._create_shortcuts_table(),
                box=PANEL_BOX,
                border_style="#334155",
                padding=(0, 1),
                title="[brand]parallax[/brand] [dim #64748b]· keyboard shortcuts[/dim #64748b]",
                title_align="left",
            )
            self.console.print()
            self.console.print(panel)
            self.console.print("[dim #64748b]Type /help commands for slash commands or /help for full guide[/dim #64748b]\n")

        elif not norm_topic or norm_topic in ("all", "guide"):
            grid = Table.grid(padding=(1, 0))
            grid.add_column()
            grid.add_row(Text("Commands", style="bold #818cf8"))
            grid.add_row(self._create_commands_table())
            grid.add_row(Text("Keyboard Shortcuts", style="bold #818cf8"))
            grid.add_row(self._create_shortcuts_table())
            grid.add_row(Text("Tip: Filter help with /help commands or /help shortcuts", style="dim #64748b"))

            panel = Panel(
                grid,
                box=PANEL_BOX,
                border_style="#334155",
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
            f"[bold #818cf8]Active Runtime Tools[/bold #818cf8] "
            f"[dim #64748b]({len(self.agent.tool_schemas)} tools available)[/dim #64748b]\n"
        )

        for schema in self.agent.tool_schemas:
            func = schema.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "").splitlines()[0] if func.get("description") else ""
            params = func.get("parameters", {}).get("properties", {})

            content = Text()
            content.append(f"{desc}\n\n", style="muted")
            content.append("Parameters: ", style="bold #94a3b8")
            if params:
                for param_name in params.keys():
                    content.append(f" {param_name} ", style="bold #93c5fd on #1e293b")
                    content.append(" ")
            else:
                content.append("none", style="dim")
            content.append("\n")

            panel = Panel(
                content,
                title=f"[bold #818cf8]{name}[/bold #818cf8]",
                title_align="left",
                box=PANEL_BOX,
                border_style="#334155",
                padding=(0, 2),
            )
            self.console.print(panel)
        self.console.print()
