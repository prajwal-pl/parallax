from __future__ import annotations

import time
from typing import Any
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
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
    ICON_SUCCESS,
    ICON_USER,
    PANEL_BOX,
)
from parallax.core.llm import extract_retry_delay
from parallax.core.types import AgentEvent, EventType, Message


class CliRenderer:
    """
    Event-driven renderer that translates an AgentEvent stream into a clean,
    modern, minimalist terminal interface without emojis.
    """

    def __init__(
        self,
        console: Console | None = None,
        verbose: bool = False,
        quiet: bool = False,
        is_interactive: bool = False,
    ):
        self.console = console or get_console()
        self.verbose = verbose
        self.quiet = quiet
        self.is_interactive = is_interactive

        self._active_status: Status | None = None
        self._run_start_time: float = 0.0
        self._tool_start_times: dict[str, float] = {}
        self._tool_actions: dict[str, str] = {}
        self._tool_calls_count: int = 0
        self._iterations: int = 0
        self._final_content: str = ""

    def start_spinner(self, status_text: str) -> None:
        """Start or update the typing/processing indicator in place."""
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

        # Only echo task in single-shot mode (interactive mode already typed it at prompt)
        if not self.is_interactive and not self.quiet:
            task = data.get("task", "")
            msg = Text()
            msg.append(f"{ICON_CHEVRON} ", style="user")
            msg.append(task, style="bold white")
            self.console.print()
            self.console.print(msg)
            self.console.print()

    def _on_model_started(self, data: dict[str, Any]) -> None:
        self._iterations = data.get("iteration", self._iterations)
        model = data.get("model", "")
        self.start_spinner(
            f"[dim #94a3b8]generating[/dim #94a3b8] [muted]· {model} · turn {self._iterations}[/muted]"
        )

    def _on_model_retry(self, data: dict[str, Any]) -> None:
        # Keep retries completely inside the live spinner without noisy console line spam
        attempt = data.get("attempt", 1)
        delay = data.get("delay", 0.0)
        self.start_spinner(
            f"[dim #94a3b8]rate limit pause[/dim #94a3b8] [muted]· retrying in {delay:.1f}s (attempt {attempt})[/muted]"
        )

    def _on_model_completed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        content = data.get("content")
        tool_count = data.get("tool_calls_count", 0)

        # Show preliminary assistant thoughts only in verbose mode to avoid terminal clutter
        if self.verbose and content and tool_count > 0 and not self.quiet:
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
        self._tool_actions[tool_call_id] = summary
        self.start_spinner(f"[dim #94a3b8]running[/dim #94a3b8] [tool.name]{summary}[/tool.name]")

    def _on_tool_completed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        tool_call_id = data.get("tool_call_id", "")
        tool_name = data.get("tool_name", "")
        result = data.get("result", "")
        start_time = self._tool_start_times.pop(tool_call_id, time.time())
        self._tool_actions.pop(tool_call_id, tool_name)
        elapsed = max(0.0, time.time() - start_time)

        if self.quiet:
            return

        receipt = Text()
        receipt.append(f"  {ICON_SUCCESS} ", style="status.success")
        receipt.append(tool_name, style="tool.name")

        if self.verbose:
            receipt.append(f" [muted]· {elapsed:.2f}s[/muted]", style="muted")
            self.console.print(receipt)
            self._render_tool_output(result)
        else:
            summary = self._extract_concise_result(tool_name, result)
            receipt.append(f" [muted]· {summary} · {elapsed:.2f}s[/muted]", style="muted")
            self.console.print(receipt)

    def _on_tool_failed(self, data: dict[str, Any]) -> None:
        self.stop_spinner()
        tool_call_id = data.get("tool_call_id", "")
        tool_name = data.get("tool_name", "")
        error = data.get("error", "Unknown error")
        start_time = self._tool_start_times.pop(tool_call_id, time.time())
        self._tool_actions.pop(tool_call_id, tool_name)
        elapsed = max(0.0, time.time() - start_time)

        if self.quiet:
            return

        receipt = Text()
        receipt.append(f"  {ICON_ERROR} ", style="status.error")
        receipt.append(tool_name, style="tool.name")

        if self.verbose:
            receipt.append(f" [status.error]failed · {elapsed:.2f}s[/status.error]", style="status.error")
            self.console.print(receipt)

            err_text = Text()
            for line in error.strip().splitlines()[:6]:
                err_text.append(f"    {ICON_LINE_V} {line}\n", style="status.error")
            self.console.print(err_text)
        else:
            clean_err = self._abstract_tool_error(error)
            receipt.append(f" [status.error]failed · {clean_err} · {elapsed:.2f}s[/status.error]", style="status.error")
            self.console.print(receipt)

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

        if self._final_content:
            self.console.print()
            role_header = Text()
            role_header.append(f"{ICON_AGENT} ", style="brand")
            role_header.append("parallax", style="brand")
            self.console.print(role_header)
            self.console.print()
            self.console.print(Markdown(self._final_content))
            self.console.print()

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
        raw_error = data.get("error", "Unknown error")
        duration = max(0.0, time.time() - self._run_start_time)

        self._render_abstracted_error(raw_error, duration)

    def _render_abstracted_error(self, raw_error: str, duration: float) -> None:
        """Abstract raw provider tracebacks into clean, high-signal actionable cards."""
        lower_err = raw_error.lower()

        if "tokens per day" in lower_err or "(tpd)" in lower_err:
            title = "Daily Token Quota Reached"
            summary = "This model's daily token quota on your provider plan is used up."
            wait = extract_retry_delay(RuntimeError(raw_error))
            if wait:
                mins = max(1, round(wait / 60))
                summary += f" It resets in about {mins} min." if mins < 120 else f" It resets in about {mins // 60} h."
            suggestions = [
                ("Switch model", "/model openai/gpt-oss-120b or /model openrouter"),
                ("Upgrade", "console.groq.com/settings/billing for higher limits"),
            ]
        elif any(k in lower_err for k in (
            "tool call validation failed", "output_parse_failed", "tool_use_failed", "failed_generation",
        )):
            title = "Malformed Tool Call"
            summary = "The model repeatedly produced an invalid tool call, even after automatic retries."
            suggestions = [
                ("Retry", "Send your request again; this is usually transient"),
                ("Switch model", "/model openai/gpt-oss-120b"),
            ]
        elif "decommissioned" in lower_err:
            title = "Model Decommissioned"
            summary = "The selected model has been decommissioned by the provider and is no longer available."
            suggestions = [
                ("Active Groq models", "/model openai/gpt-oss-120b or /model qwen/qwen3.8-27b"),
                ("Switch provider", "/model openrouter"),
            ]
        elif "model_not_found" in lower_err or "does not exist or you do not have access" in lower_err:
            title = "Model Not Accessible"
            summary = "The requested model is not available or not enabled for your account/API tier."
            suggestions = [
                ("Browse available models", "/model groq or /model openrouter"),
                ("Run doctor", "parallax doctor to verify API keys and connectivity"),
            ]
        elif "tool calling is not supported" in lower_err or "does not support chat" in lower_err:
            title = "Model Incompatible"
            summary = "This model does not support chat completions or function calling required by the agent."
            suggestions = [
                ("Recommended Groq models", "/model openai/gpt-oss-120b or /model qwen/qwen3.8-27b"),
                ("Recommended OpenRouter", "/model liquid/lfm-2.5-2.6b:free"),
            ]
        elif any(k in lower_err for k in ("tpm", "tokens per minute", "413", "request too large", "rate_limit_exceeded")):
            title = "Token Rate Limit Exceeded"
            summary = "The provider tokens-per-minute (TPM) quota was exceeded. History can be compacted to reduce token volume."
            suggestions = [
                ("Wait a moment", "Groq rate limits reset on a 60-second sliding window"),
                ("Compact context", "/compact to summarize earlier conversation turns"),
                ("Switch to high-capacity model", "/model openai/gpt-oss-120b"),
                ("Switch to OpenRouter", "/model openrouter"),
            ]
        elif "free-models-per-day" in lower_err:
            title = "Provider Quota Limit"
            summary = "The daily free request quota for this model on OpenRouter has been reached."
            suggestions = [
                ("Switch model", "/model meta-llama/llama-3.3-70b-instruct:free"),
                ("Add credits", "openrouter.ai/credits to unlock higher rate limits"),
            ]
        elif "rate limit" in lower_err or "429" in lower_err:
            title = "Provider Rate Limit"
            summary = "The request limit or quota for this model has been reached."
            suggestions = [
                ("Switch model", "/model meta-llama/llama-3.3-70b-instruct:free or /model groq"),
                ("Compact context", "/compact to reduce tokens sent per request"),
            ]
        elif "unauthorized" in lower_err or "401" in lower_err or "invalid api key" in lower_err:
            title = "Authentication Failure"
            summary = "The active model provider's API key is missing or invalid."
            suggestions = [
                ("Switch to Groq", "If using GROQ_API_KEY, switch to a Groq model: '/model groq' or '--model llama-3.3-70b-versatile'"),
                ("Switch to OpenRouter", "If using OPENROUTER_API_KEY, check your key or switch with '/model openrouter'"),
                ("Configure key", "Ensure the matching GROQ_API_KEY or OPENROUTER_API_KEY is configured in .env"),
                ("Diagnostics", "Run 'parallax doctor' to inspect API key configuration"),
            ]
        elif "context length" in lower_err or "maximum context" in lower_err:
            title = "Context Limit Exceeded"
            summary = "The conversation history exceeds the model's token capacity."
            suggestions = [
                ("Compact history", "/compact to summarize older conversation turns"),
                ("Clear session", "/clear to start a fresh session context"),
            ]
        elif "timeout" in lower_err or "timed out" in lower_err:
            title = "Connection Timeout"
            summary = "The network request to the model provider timed out."
            suggestions = [
                ("Retry request", "Send your task again or verify network connectivity"),
            ]
        else:
            title = "Execution Failed"
            clean_msg = raw_error.strip().split("\n")[0]
            if len(clean_msg) > 120:
                clean_msg = clean_msg[:117] + "..."
            summary = f"Provider returned an error: {clean_msg}" if clean_msg else "The agent encountered an unrecoverable provider failure."
            suggestions = [
                ("Diagnostics", "Run 'parallax doctor' to inspect environment health"),
                ("Switch model", "Run '/model' to select another available provider model"),
            ]

        content = Text()
        content.append(f"{summary}\n\n", style="#f87171")
        content.append("Suggestions:\n", style="bold #94a3b8")
        for label, action in suggestions:
            content.append(f"  • {label}: ", style="muted")
            content.append(f"{action}\n", style="tool.name")

        if self.verbose:
            content.append(f"\nTechnical details: {raw_error}\n", style="dim #64748b")

        self.console.print()
        panel = Panel(
            content,
            title=f"[bold #f87171]{ICON_ERROR} {title} ({duration:.1f}s)[/bold #f87171]",
            title_align="left",
            box=PANEL_BOX,
            border_style="#ef4444",
            padding=(0, 2),
        )
        self.console.print(panel)
        self.console.print()

    def _format_tool_action(self, name: str, args: dict[str, Any]) -> str:
        """Format tool call arguments into a concise one-line summary."""
        if name == "bash":
            cmd = args.get("command", "")
            return f"bash $ {cmd[:60]}..." if len(cmd) > 60 else f"bash $ {cmd}"
        elif name in ("editor", "str_replace_editor"):
            action = args.get("command", "")
            path = args.get("path", "")
            return f"{action} {path}" if action and path else f"editor {action or path}"
        elif name in ("search", "search_code"):
            query = args.get("query", "")
            return f'search "{query[:40]}"'
        arg_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
        return f"{name}({arg_str})" if arg_str else name

    def _abstract_tool_error(self, error: str) -> str:
        """Extract a clean, concise one-line summary of a tool error without spamming the screen."""
        if not error:
            return "unknown error"
        line = error.strip().splitlines()[0].strip()
        prefix = "Something went wrong, tool execution failed with error: "
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
        if line.startswith("ToolError: "):
            line = line[len("ToolError: "):].strip()
        line = line.replace("[Errno 22] Invalid argument", "invalid argument")
        if len(line) > 60:
            line = line[:57] + "..."
        return line or "unknown error"

    def _extract_concise_result(self, tool_name: str, result: str) -> str:
        """Extract a concise summary from tool output for non-verbose display."""
        if not result or result.strip() in ("", "None"):
            return "completed"
        trimmed = result.strip()
        lines = trimmed.splitlines()

        if len(lines) == 1 and len(trimmed) <= 45:
            return trimmed

        if tool_name in ("search", "search_code"):
            for line in lines:
                if "Found " in line and "matches" in line:
                    return line.strip()
                if "No matches found" in line:
                    return "no matches"
            return f"{len(lines)} results"

        if tool_name == "bash":
            for line in reversed(lines):
                if any(k in line.lower() for k in ("passed", "failed", "error", "exit code")):
                    if len(line.strip()) <= 45:
                        return line.strip()
            return f"completed ({len(lines)} lines)"

        if tool_name in ("editor", "str_replace_editor"):
            return "completed"

        return "completed"

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
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(f"    [muted]{ICON_LINE_V}[/muted] [diff.add]{line}[/diff.add]")
            elif line.startswith("-") and not line.startswith("---"):
                self.console.print(f"    [muted]{ICON_LINE_V}[/muted] [diff.remove]{line}[/diff.remove]")
            else:
                self.console.print(f"    [muted]{ICON_LINE_V}[/muted] [muted]{line}[/muted]")

    def render_history(self, messages: list[Message]) -> None:
        """
        Replay past conversation messages visually on the terminal.
        Renders user turns, compact tool receipts, and assistant responses.
        """
        if not messages:
            return

        rendered_any = False
        for msg in messages:
            if msg.role == "system":
                continue

            if msg.role == "user":
                self.console.print()
                user_line = Text()
                user_line.append(f"{ICON_USER} ", style="brand")
                user_line.append(msg.content, style="bold white")
                self.console.print(user_line)
                rendered_any = True

            elif msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        receipt = Text()
                        receipt.append(f"  {ICON_SUCCESS} ", style="status.success")
                        receipt.append(tc.name, style="tool.name")
                        if tc.arguments:
                            arg_hint = ""
                            for k in ("path", "command", "query", "url", "target"):
                                if k in tc.arguments:
                                    arg_hint = str(tc.arguments[k])
                                    break
                            if not arg_hint:
                                first_val = next(iter(tc.arguments.values()), None)
                                if first_val and isinstance(first_val, (str, int, float)):
                                    arg_hint = str(first_val)
                            if arg_hint:
                                if len(arg_hint) > 40:
                                    arg_hint = arg_hint[:37] + "..."
                                receipt.append(f" [muted]· {arg_hint}[/muted]", style="muted")
                        self.console.print(receipt)
                    rendered_any = True

                if msg.content:
                    self.console.print()
                    role_header = Text()
                    role_header.append(f"{ICON_AGENT} ", style="brand")
                    role_header.append("parallax", style="brand")
                    self.console.print(role_header)
                    self.console.print()
                    self.console.print(Markdown(msg.content))
                    self.console.print()
                    rendered_any = True

            elif msg.role == "tool":
                # Intermediate raw tool execution results are abstracted via tool receipts above
                pass

        if rendered_any:
            self.console.rule("[dim #64748b]history restored[/dim #64748b]", style="border")
            self.console.print()
