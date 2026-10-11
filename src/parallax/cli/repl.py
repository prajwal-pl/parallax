from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from rich.console import Console
from rich.table import Table
from rich.text import Text

from parallax.cli.console import get_console, print_hero
from parallax.cli.renderer import CliRenderer
from parallax.cli.theme import (
    ICON_BULLET,
    ICON_CHEVRON,
    ICON_ERROR,
    ICON_SUCCESS,
    PANEL_BOX,
)
from parallax.core.context.compactor import compact
from parallax.core.types import EventType, Message


if TYPE_CHECKING:
    from parallax.core.agent import Agent


class InteractiveRepl:
    """
    Interactive terminal REPL providing a multi-turn conversation loop,
    slash commands, and live event rendering.
    """

    def __init__(self, agent: Agent, console: Console | None = None, verbose: bool = False):
        self.agent = agent
        self.console = console or get_console()
        self.verbose = verbose
        self.messages: list[Message] = []

    async def run(self) -> None:
        """Run the interactive REPL loop."""
        print_hero(model=self.agent.model)

        while True:
            try:
                # Clean prompt indicator
                prompt_text = f"[user]{ICON_CHEVRON}[/user] "
                user_input = self.console.input(prompt_text).strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[muted]session ended[/muted]")
                break

            if not user_input:
                continue

            # Check for slash commands
            if user_input.startswith("/"):
                handled = await self._handle_command(user_input)
                if handled == "exit":
                    self.console.print("[muted]session ended[/muted]")
                    break
                continue

            # Execute conversational turn
            await self._run_turn(user_input)

    async def _run_turn(self, user_input: str) -> None:
        """Run a single agent turn with live event streaming and history persistence."""
        renderer = CliRenderer(console=self.console, verbose=self.verbose)
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
            self.console.print(f"\n  [status.error]{ICON_ERROR} turn failed: {e}[/status.error]\n")

    async def _handle_command(self, cmd_line: str) -> str | None:
        """Handle slash command execution."""
        parts = cmd_line.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command in ("/exit", "/quit"):
            return "exit"

        elif command == "/help":
            self._show_help()

        elif command == "/clear":
            self.messages.clear()
            self.console.print(f"[status.success]{ICON_SUCCESS}[/status.success] [muted]conversation history cleared[/muted]\n")

        elif command == "/model":
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

        elif command == "/context":
            self._show_context()

        elif command == "/compact":
            self._compact_context()

        elif command == "/tools":
            self._show_tools()

        else:
            self.console.print(
                f"[status.error]{ICON_ERROR}[/status.error] [muted]unknown command '{command}'. Type /help for available commands.[/muted]\n"
            )

        return None

    def _show_help(self) -> None:
        table = Table(
            box=PANEL_BOX,
            border_style="border",
            show_header=True,
            header_style="bold #818cf8",
            padding=(0, 1),
        )
        table.add_column("command", style="brand", no_wrap=True)
        table.add_column("description", style="muted")

        table.add_row("/help", "Show this help menu")
        table.add_row("/model [name]", "View or switch active model")
        table.add_row("/context", "Inspect conversation turns and token count")
        table.add_row("/compact", "Manually compact conversation history")
        table.add_row("/tools", "List all registered runtime tools")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/exit, /quit", "Exit the interactive session")

        self.console.print(table)
        self.console.print()

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
        table = Table(
            box=PANEL_BOX,
            border_style="border",
            show_header=True,
            header_style="bold #818cf8",
            padding=(0, 1),
        )
        table.add_column("tool", style="tool.name", no_wrap=True)
        table.add_column("parameters", style="muted")

        for schema in self.agent.tool_schemas:
            func = schema.get("function", {})
            name = func.get("name", "")
            params = func.get("parameters", {}).get("properties", {})
            param_names = ", ".join(params.keys()) if params else "none"
            table.add_row(name, param_names)

        self.console.print(table)
        self.console.print()
