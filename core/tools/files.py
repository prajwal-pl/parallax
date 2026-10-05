from pathlib import Path
import os

from core.errors import ToolError
from core.tool import tool

IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
MAX_LIST_RESULTS = 500


class FileTools:
    def __init__(self, workspace: str):
        self.workspace = workspace

    def validate_path(self, path: str) -> Path:
        workspace_path = Path(self.workspace).resolve()
        file_path = (workspace_path / path).resolve()
        if not file_path.is_relative_to(workspace_path):
            raise ToolError(f"Permission denied: path '{path}' is outside the workspace.")
        return file_path

    @tool
    def read_file(self, path: str) -> str:
        """
        Read the full text contents of a file.

        Args:
            path: Relative path to the file within the workspace.
        """
        resolved = self.validate_path(path)
        if not resolved.exists():
            raise ToolError(f"File not found: {path}")
        if not resolved.is_file():
            raise ToolError(f"Path is a directory, not a file: {path}")
        return resolved.read_text(encoding="utf-8")

    @tool
    def write_file(self, path: str, content: str) -> str:
        """
        Create a new file or completely overwrite an existing file.
        To modify a specific section of an existing file, use edit_file instead.

        Args:
            path: Relative path to the file within the workspace.
            content: Complete text content to write.
        """
        resolved = self.validate_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    @tool
    def list_files(self, path: str = ".") -> str:
        """
        List files and directories in the workspace or a specific subdirectory.

        Args:
            path: Directory path to inspect (defaults to workspace root).
        """
        root = self.validate_path(path)
        if not root.is_dir():
            raise ToolError(f"Directory not found: {path}")

        workspace_path = Path(self.workspace).resolve()
        files: list[str] = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                full = Path(dirpath) / name
                files.append(str(full.relative_to(workspace_path)))
                if len(files) >= MAX_LIST_RESULTS:
                    return (
                        "\n".join(files)
                        + f"\n... showing first {MAX_LIST_RESULTS} files, list a subdirectory to see more"
                    )

        return "\n".join(files) if files else "No files found"

    @tool
    def edit_file(self, path: str, old: str, new: str) -> str:
        """
        Replace one exact occurrence of old with new in an existing file.
        The old string must appear exactly once in the file to avoid ambiguity.

        Args:
            path: Relative path to the file within the workspace.
            old: The exact verbatim string to find and replace.
            new: The replacement string.
        """
        resolved = self.validate_path(path)
        if not resolved.exists():
            raise ToolError(f"File not found: {path}")
        if not resolved.is_file():
            raise ToolError(f"Path is a directory, not a file: {path}")

        text = resolved.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolError(
                f"Text to replace was not found in {path}. "
                "Ensure exact match including whitespace and indentation."
            )
        if count > 1:
            raise ToolError(
                f"Found {count} occurrences of text to replace in {path}. "
                "Include more surrounding context lines in 'old' to make it unique."
            )

        resolved.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"Successfully edited {path}"
