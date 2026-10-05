from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.errors import ToolError

IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int


class Environment(Protocol):
    """
    Abstract environment contract for filesystem access and process execution.
    Decouples tools from the host operating system.
    """

    def read_file(self, path: str) -> str:
        """Read full text content of a file."""
        ...

    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite a file."""
        ...

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        ...

    def is_file(self, path: str) -> bool:
        """Check if a path is a regular file."""
        ...

    def is_dir(self, path: str) -> bool:
        """Check if a path is a directory."""
        ...

    def list_dir(self, path: str, max_depth: int = 2) -> tuple[list[str], int]:
        """
        List files and subdirectories up to max_depth.
        Returns:
            (list_of_relative_paths, omitted_hidden_count)
        """
        ...

    def execute(self, command: str, timeout: float = 30.0) -> ExecResult:
        """Execute a shell command inside the environment."""
        ...


class LocalEnvironment:
    """
    Concrete environment running against a local workspace directory.
    Enforces path sandboxing and clean process-group termination.
    """

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        if not self.workspace.exists():
            self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """Resolves path and enforces that it does not escape the workspace."""
        resolved = (self.workspace / path).resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ToolError(f"Permission denied: path '{path}' escapes workspace boundary.")
        return resolved

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"File not found: {path}")
        if not target.is_file():
            raise ToolError(f"Path is a directory, not a file: {path}")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"File '{path}' appears to be binary and cannot be read as text.")

    def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def is_file(self, path: str) -> bool:
        return self._resolve(path).is_file()

    def is_dir(self, path: str) -> bool:
        return self._resolve(path).is_dir()

    def list_dir(self, path: str, max_depth: int = 2) -> tuple[list[str], int]:
        target = self._resolve(path)
        if not target.is_dir():
            raise ToolError(f"Directory not found: {path}")

        entries: list[str] = []
        hidden_count = 0
        target_depth = len(target.parts)

        for root, dirs, files in os.walk(target):
            curr_depth = len(Path(root).parts) - target_depth
            if curr_depth >= max_depth:
                dirs.clear()
                continue

            hidden_dirs = [d for d in dirs if d.startswith(".") or d in IGNORED_DIRS]
            hidden_count += len(hidden_dirs)
            dirs[:] = sorted([d for d in dirs if d not in hidden_dirs])

            for d in dirs:
                rel = Path(root, d).relative_to(self.workspace)
                entries.append(f"{rel}/")

            for f in sorted(files):
                if f.startswith("."):
                    hidden_count += 1
                else:
                    rel = Path(root, f).relative_to(self.workspace)
                    entries.append(str(rel))

        return entries, hidden_count

    def execute(self, command: str, timeout: float = 30.0) -> ExecResult:
        process = subprocess.Popen(
            command,
            shell=True,
            text=True,
            cwd=str(self.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            stdout, stderr = process.communicate()
            raise ToolError(f"Command timed out after {timeout} seconds.")

        return ExecResult(stdout=stdout, stderr=stderr, returncode=process.returncode)
