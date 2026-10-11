from __future__ import annotations

import time
from typing import Any
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.text import Text

from parallax.cli.console import get_console
from parallax.cli.theme import (
    ICON_AGENT,
    ICON_ARROW,
    ICON_BULLET,
    ICON_CHEVRON,
    ICON_COMPACT,
    ICON_ERROR,
    ICON_LINE_V,
    ICON_RETRY,
    ICON_SUCCESS,
    PANEL_BOX,
)
from parallax.core.types import AgentEvent, EventType


class CliRenderer:
    """
    Event-driven renderer that translates an AgentEvent stream into a clean,
    modern, minimalist terminal interface without emojis.
    """

    def __init__(self, console: Console | None = None, verbose: bool = False, quiet: bool = False):
        self.console = console or get_console()
        self.verbose = verbose
        self.quiet = quiet

        self._active_status: Status | None = None
        self._run_start_time: float = 0.0
        self._tool_start_times: dict[str, float] = {}
        self._tool_calls_count: int = 0
        self._iterations: int = 0
        self._final_content: str = ""

    def start_spinner(self, status_text: str) -> None:
        """Start or update the typing/processing indicator."""
        if self.quiet:
            return
        if self._active_status is None:
            self._active_status = self.console.status(
                status_text,
                spinner="dots",
                spinner_style="#818cf8",
            )
            self._active_status.start()
        else:
            self._active_status.update(status_text)

    def stop_spinner(self) -> None:
        """Stop the active indicator if running."""
        if self._active_status is not None:
            self._active_status.stop()
            self._active_status = None

    def handle_event(self, event: AgentEvent) -> None:
        """Dispatch an AgentEvent to its corresponding render method."""
        handlers = {
            EventType.RUN_STARTED: self._on_run_started,
            EventType.MODEL_STARTED: self._on_model_started,
            EventType.MODEL_RETRY: self._on_model_retry,
            EventType.MODEL_COMPLETED: self._on_model_completed,
            EventType.TOOL_STARTED: self._on_tool_started,
            EventType.TOOL_COMPLETED: self._on_tool_completed,
            EventType.TOOL_FAILED: self._on_tool_failed,
            EventType.CONTEXT_COMPACTED: self._on_context_compacted,
            EventType.RUN_COMPLETED: self._on_run_completed,
            EventType.RUN_FAILED: self._on_run_failed,
        }
        handler = handlers.get(event.type)
        if handler:
            handler(event.data)

    def _on_run_started(self, data: dict[str, Any]) -> None:
        self._run_start_time = time.time()
        self._tool_calls_count = 0
        self._iterations = 0
        self._final_content = ""

        if not self.quiet:
            task = data.get("task", "")
            # Print subtle task marker
            msg = Text()
            msg.append(f"{ICON_CHEVRON} ", style="user")
            msg.append(task, style="bold white")
            self.console.print()
            self.console.print(msg)
            self.console.print()

    def _on_model_started(self, data: dict[str, Any]) -> None:
        self._iterations = data.get("iteration", self._iterations)
        model = data.get("model", "")
        # Animated indicator without emojis
        self.start_spinner(f"[dim #94a3b8]generating[/dim #94a3b8] [muted]· {model} · iteration {self._iterations}[/muted]")

    def _on_model_retry(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        attempt = data.get("attempt", 1)
        delay = data.get("delay", 0.0)
        error = data.get("error", "Transient error")

        # Clean warning notice
        msg = Text()
        msg.append(f"  {ICON_RETRY} ", style="status.warning")
        msg.append(f"rate limit / network pause ({delay:.1f}s) ", style="status.warning")
        msg.append(f"{ICON_BULLET} attempt {attempt} {ICON_BULLET} {error}", style="muted")
        self.console.print(msg)

        # Resume spinner
        self.start_spinner("[dim #94a3b8]waiting to retry[/dim #94a3b8]")

    def _on_model_completed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        content = data.get("content")
        tool_count = data.get("tool_calls_count", 0)

        # If model generated intermediate text before calling tools, show it subtly
        if content and tool_count > 0 and not self.quiet:
            for line in content.strip().splitlines():
                if line.strip():
                    self.console.print(f"  [muted]{ICON_LINE_V}[/muted] [dim]{line}[/dim]")

    def _on_tool_started(self, data: dict[str, Any]) -> None:
        tool_call_id = data.get("tool_call_id", "")
        tool_name = data.get("tool_name", "")
        arguments = data.get("arguments", {})
        self._tool_start_times[tool_call_id] = time.time()
        self._tool_calls_count += 1

        summary = self._format_tool_action(tool_name, arguments)
        self.start_spinner(f"[dim #94a3b8]executing[/dim #94a3b8] [tool.name]{summary}[/tool.name]")

    def _on_tool_completed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        tool_call_id = data.get("tool_call_id", "")
        tool_name = data.get("tool_name", "")
        result = data.get("result", "")
        start_time = self._tool_start_times.pop(tool_call_id, time.time())
        elapsed = max(0.0, time.time() - start_time)

        if self.quiet:
            return

        # Success receipt line
        receipt = Text()
        receipt.append(f"  {ICON_SUCCESS} ", style="status.success")
        receipt.append(tool_name, style="tool.name")
        receipt.append(f" {ICON_BULLET} {elapsed:.2f}s", style="muted")
        self.console.print(receipt)

        # Show bounded output preview
        self._render_tool_output(result)

    def _on_tool_failed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        tool_call_id = data.get("tool_call_id", "")
        tool_name = data.get("tool_name", "")
        error = data.get("error", "Unknown error")
        start_time = self._tool_start_times.pop(tool_call_id, time.time())
        elapsed = max(0.0, time.time() - start_time)

        if self.quiet:
            return

        receipt = Text()
        receipt.append(f"  {ICON_ERROR} ", style="status.error")
        receipt.append(tool_name, style="tool.name")
        receipt.append(f" {ICON_BULLET} failed {ICON_BULLET} {elapsed:.2f}s", style="status.error")
        self.console.print(receipt)

        # Render error box indented
        err_text = Text()
        for line in error.strip().splitlines()[:6]:
            err_text.append(f"    {ICON_LINE_V} {line}\n", style="status.error")
        self.console.print(err_text)

    def _on_context_compacted(self, data: dict[str, Any]) -> None:
        if self.quiet:
            return
        tokens_before = data.get("tokens_before", 0)
        tokens_after = data.get("tokens_after", 0)
        dropped_turns = data.get("dropped_turns", 0)

        notice = Text()
        notice.append(f"  {ICON_COMPACT} ", style="brand")
        notice.append("context compacted: ", style="muted")
        notice.append(f"{tokens_before:,} {ICON_ARROW} {tokens_after:,} tokens ", style="tool.name")
        notice.append(f"({dropped_turns} turns summarized)", style="muted")
        self.console.print(notice)

    def _on_run_completed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        duration = max(0.0, time.time() - self._run_start_time)
        self._final_content = data.get("content", "")

        # Render final markdown response
        if self._final_content:
            self.console.print()
            role_header = Text()
            role_header.append(f"{ICON_AGENT} ", style="brand")
            role_header.append("parallax", style="brand")
            self.console.print(role_header)
            self.console.print()
            self.console.print(Markdown(self._final_content))
            self.console.print()

        # Render sleek execution summary footer
        if not self.quiet:
            summary = (
                f"completed in {duration:.1f}s {ICON_BULLET} "
                f"{self._iterations} iterations {ICON_BULLET} "
                f"{self._tool_calls_count} tool calls"
            )
            self.console.rule(f"[muted]{summary}[/muted]", style="border")
            self.console.print()

    def _on_run_failed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        error = data.get("error", "Unknown error")
        duration = max(0.0, time.time() - self._run_start_time)

        self.console.print()
        panel = Panel(
            Text(error, style="status.error"),
            title=f"[status.error]{ICON_ERROR} run failed ({duration:.1f}s)[/status.error]",
            title_align="left",
            box=PANEL_BOX,
            border_style="status.error",
            padding=(0, 1),
        )
        self.console.print(panel)
        self.console.print()

    def _format_tool_action(self, name: str, args: dict[str, Any]) -> str:
        """Format tool call arguments into a concise one-line summary."""
        if name == "bash":
            cmd = args.get("command", "")
            return f"bash $ {cmd[:60]}..." if len(cmd) > 60 else f"bash $ {cmd}"
        elif name == "editor":
            action = args.get("command", "")
            path = args.get("path", "")
            return f"editor {action} {path}"
        elif name == "search":
            query = args.get("query", "")
            return f'search "{query[:40]}"'
        # Generic tool fallback
        arg_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
        return f"{name}({arg_str})"

    def _render_tool_output(self, output: str) -> None:
        """Render a clean bounded preview of tool output."""
        if not output or output.strip() in ("", "None"):
            return

        lines = output.strip().splitlines()
        max_lines = 16 if self.verbose else 8

        if len(lines) <= max_lines:
            display_lines = lines
            omitted = 0
        else:
            head_count = max_lines // 2
            tail_count = max_lines // 2
            display_lines = lines[:head_count]
            omitted = len(lines) - head_count - tail_count
            display_lines.append(f"[dim]... {omitted} lines omitted ...[/dim]")
            display_lines.extend(lines[-tail_count:])

        for line in display_lines:
            # Color diff lines if present
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(f"    [muted]{ICON_LINE_V}[/muted] [diff.add]{line}[/diff.add]")
            elif line.startswith("-") and not line.startswith("---"):
                self.console.print(f"    [muted]{ICON_LINE_V}[/muted] [diff.remove]{line}[/diff.remove]")
            else:
                self.console.print(f"    [muted]{ICON_LINE_V}[/muted] [muted]{line}[/muted]")
