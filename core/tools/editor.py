from __future__ import annotations

import re
from typing import Literal

from core.environment import Environment
from core.errors import ToolError
from core.tool import tool

SNIPPET_CONTEXT_WINDOW = 4
MAX_VIEW_LINES = 1000


class EditorTool:
    """
    Production filesystem editor tool modeled after Anthropic & OpenHands ACI.
    Interacts purely through an abstract Environment.
    """

    def __init__(self, env: Environment):
        self.env = env
        # History stack for undo: {path: [previous_file_contents]}
        self._history: dict[str, list[str]] = {}

    def _format_lines(self, lines: list[str], start_line: int = 1) -> str:
        """Formats code lines with standard line numbers (e.g. cat -n format)."""
        return "\n".join(f"{i + start_line:6}\t{line}" for i, line in enumerate(lines))

    def _push_history(self, path: str, content: str) -> None:
        stack = self._history.setdefault(path, [])
        stack.append(content)
        if len(stack) > 10:
            stack.pop(0)

    # -------------------------------------------------------------------------
    # Command Implementations
    # -------------------------------------------------------------------------

    def _view(self, path: str, view_range: list[int] | None) -> str:
        # Directory View
        if self.env.is_dir(path):
            if view_range:
                raise ToolError("view_range is not supported when path is a directory.")
            entries, hidden_count = self.env.list_dir(path, max_depth=2)
            output = f"Files and directories up to 2 levels deep in '{path}':\n"
            output += "\n".join(entries) if entries else "(empty directory)"
            if hidden_count > 0:
                output += f"\n\n({hidden_count} hidden/ignored items omitted)"
            return output

        # File View
        content = self.env.read_file(path)
        lines = content.splitlines()
        total_lines = len(lines)

        if not view_range:
            display_lines = lines[:MAX_VIEW_LINES]
            result = self._format_lines(display_lines, start_line=1)
            if total_lines > MAX_VIEW_LINES:
                result += f"\n\n[Showing first {MAX_VIEW_LINES} of {total_lines} lines. Use view_range to inspect remaining lines.]"
            return result

        if len(view_range) != 2:
            raise ToolError(
                "view_range must be a list of two integers: [start_line, end_line]."
            )

        start, end = view_range
        if start < 1 or start > max(1, total_lines):
            raise ToolError(f"start_line ({start}) must be within [1, {total_lines}].")

        if end == -1 or end > total_lines:
            end = total_lines

        if end < start:
            raise ToolError(
                f"end_line ({end}) cannot be less than start_line ({start})."
            )

        selected = lines[start - 1 : end]
        return (
            f"File '{path}' lines {start}-{end} of {total_lines}:\n"
            + self._format_lines(selected, start_line=start)
        )

    def _create(self, path: str, file_text: str | None) -> str:
        if file_text is None:
            raise ToolError("Parameter 'file_text' is required when command='create'.")
        if self.env.exists(path):
            raise ToolError(
                f"File already exists at '{path}'. Cannot overwrite using 'create'. "
                "Use 'str_replace' to edit existing files."
            )

        self.env.write_file(path, file_text)
        self._push_history(path, "")
        return f"File created successfully at '{path}' ({len(file_text)} characters)."

    def _str_replace(self, path: str, old_str: str | None, new_str: str | None) -> str:
        if old_str is None:
            raise ToolError(
                "Parameter 'old_str' is required when command='str_replace'."
            )
        new_str = new_str or ""
        if old_str == new_str:
            raise ToolError("'new_str' must be different from 'old_str'.")

        content = self.env.read_file(path)

        # Exact match pattern
        pattern = re.escape(old_str)
        matches = list(re.finditer(pattern, content))

        if not matches:
            # Fallback heuristic: check if stripped whitespace would have matched
            stripped_matches = list(re.finditer(re.escape(old_str.strip()), content))
            if stripped_matches:
                raise ToolError(
                    f"old_str was not found verbatim, but matches when leading/trailing whitespace is stripped. "
                    "Ensure exact indentation and newline match."
                )
            raise ToolError(
                f"old_str was not found in '{path}'. Ensure exact verbatim match."
            )

        if len(matches) > 1:
            line_numbers = [content[: m.start()].count("\n") + 1 for m in matches]
            raise ToolError(
                f"Multiple occurrences ({len(matches)}) of old_str found at lines {line_numbers}. "
                "Include more surrounding context lines in old_str to ensure a unique match."
            )

        match = matches[0]
        start_idx, end_idx = match.start(), match.end()

        # Save current state for undo
        self._push_history(path, content)

        new_content = content[:start_idx] + new_str + content[end_idx:]
        self.env.write_file(path, new_content)

        # Compute preview snippet of the edited area
        replaced_line = content[:start_idx].count("\n") + 1
        new_lines = new_content.splitlines()
        snip_start = max(1, replaced_line - SNIPPET_CONTEXT_WINDOW)
        snip_end = min(
            len(new_lines),
            replaced_line + new_str.count("\n") + SNIPPET_CONTEXT_WINDOW,
        )

        snippet = self._format_lines(
            new_lines[snip_start - 1 : snip_end], start_line=snip_start
        )
        return (
            f"The file '{path}' has been edited.\n"
            f"Preview of changes:\n{snippet}\n"
            "Review the changes and verify they are as expected."
        )

    def _insert(self, path: str, insert_line: int | None, new_str: str | None) -> str:
        if insert_line is None:
            raise ToolError(
                "Parameter 'insert_line' is required when command='insert'."
            )
        if new_str is None:
            raise ToolError("Parameter 'new_str' is required when command='insert'.")

        content = self.env.read_file(path)
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        if insert_line < 0 or insert_line > total_lines:
            raise ToolError(
                f"insert_line ({insert_line}) must be between 0 and {total_lines}."
            )

        self._push_history(path, content)

        insert_text = new_str if new_str.endswith("\n") else new_str + "\n"
        lines.insert(insert_line, insert_text)

        new_content = "".join(lines)
        self.env.write_file(path, new_content)

        all_lines = new_content.splitlines()
        snip_start = max(1, insert_line - SNIPPET_CONTEXT_WINDOW + 1)
        snip_end = min(
            len(all_lines),
            insert_line + new_str.count("\n") + SNIPPET_CONTEXT_WINDOW + 1,
        )
        snippet = self._format_lines(
            all_lines[snip_start - 1 : snip_end], start_line=snip_start
        )

        return (
            f"Inserted text after line {insert_line} in '{path}'.\n"
            f"Preview:\n{snippet}\n"
            "Review the changes and verify they are as expected."
        )

    def _undo_edit(self, path: str) -> str:
        stack = self._history.get(path, [])
        if not stack:
            raise ToolError(f"No edit history available to undo for '{path}'.")

        previous_content = stack.pop()
        self.env.write_file(path, previous_content)
        return f"Successfully reverted '{path}' to its previous state."

    # -------------------------------------------------------------------------
    # Public Tool Method
    # -------------------------------------------------------------------------

    @tool
    def str_replace_editor(
        self,
        command: Literal["view", "create", "str_replace", "insert", "undo_edit"],
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
    ) -> str:
        """
        Custom filesystem editor tool for viewing, creating, and modifying files.

        Args:
            command: The command to run. Allowed options: view, create, str_replace, insert, undo_edit.
            path: Relative path to file or directory within workspace.
            file_text: Required for 'create'. Content of the file to create.
            view_range: Optional for 'view'. Line range [start_line, end_line] (1-indexed). Use [start, -1] for until EOF.
            old_str: Required for 'str_replace'. The exact verbatim string to replace.
            new_str: The replacement string for 'str_replace', or content to insert for 'insert'.
            insert_line: Required for 'insert'. The line number after which to insert new_str.
        """

        match command:
            case "view":
                return self._view(path, view_range)
            case "create":
                return self._create(path, file_text)
            case "str_replace":
                return self._str_replace(path, old_str, new_str)
            case "insert":
                return self._insert(path, insert_line, new_str)
            case "undo_edit":
                return self._undo_edit(path)
            case _:
                raise ToolError(
                    f"Unrecognized command: '{command}'. Allowed commands: view, create, str_replace, insert, undo_edit."
                )
