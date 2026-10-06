from __future__ import annotations

import re

from core.environment import Environment
from core.errors import ToolError
from core.tool import tool


class SearchTool:
    """
    Tool for searching code and text patterns across files in the workspace.
    """

    def __init__(self, env: Environment):
        self.env = env

    @tool
    def search_code(
        self,
        query: str,
        path: str = ".",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> str:
        """
        Search for a text string or regex pattern across files in the workspace.

        Args:
            query: The text or regex pattern to search for.
            path: Directory or file to search within (default: ".").
            case_sensitive: Whether the search should be case sensitive (default: False).
            max_results: Maximum number of matching lines to return (default: 50).
        """
        if not self.env.exists(path):
            raise ToolError(f"Path does not exist: {path}")

        # Compile pattern (fallback to literal search if query has invalid regex syntax)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags=flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags=flags)

        # Collect files to search
        files_to_search: list[str] = []
        if self.env.is_file(path):
            files_to_search.append(path)
        else:
            entries, _ = self.env.list_dir(path, max_depth=10)
            # Filter only files (directories end with '/')
            files_to_search.extend([e for e in entries if not e.endswith("/")])

        matches: list[str] = []

        for file_path in sorted(files_to_search):
            try:
                content = self.env.read_file(file_path)
            except ToolError:
                # Skip binary files or unreadable files
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{file_path}:{line_num}: {line.strip()}")
                    if len(matches) >= max_results:
                        break

            if len(matches) >= max_results:
                break

        if not matches:
            return f"No matches found for '{query}' in '{path}'."

        result = f"Found {len(matches)} match{'es' if len(matches) > 1 else ''}:\n"
        result += "\n".join(matches)
        if len(matches) >= max_results:
            result += f"\n\n... [Showing first {max_results} results. Use a more specific query or subpath.]"

        return result
