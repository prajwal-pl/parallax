from __future__ import annotations

from core.environment import Environment
from core.tool import tool

MAX_BASH_OUTPUT_CHARS = 30_000


class BashTool:
    """
    Tool for running shell commands inside the environment.
    """

    def __init__(self, env: Environment):
        self.env = env

    @tool
    def bash(self, command: str, timeout: float = 30.0) -> str:
        """
        Execute a bash command in the environment and return its output.

        Args:
            command: The shell command line to execute.
            timeout: Maximum execution time in seconds (default: 30.0).
        """
        result = self.env.execute(command, timeout=timeout)

        # Merge stdout and stderr so the model sees all output
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(result.stderr)

        combined_output = "\n".join(output_parts).strip()

        # Truncate if output exceeds safety budget
        if len(combined_output) > MAX_BASH_OUTPUT_CHARS:
            combined_output = (
                combined_output[:MAX_BASH_OUTPUT_CHARS]
                + f"\n\n... [Output truncated: exceeded {MAX_BASH_OUTPUT_CHARS} characters]"
            )

        status = (
            "succeeded (exit code 0)"
            if result.returncode == 0
            else f"failed (exit code {result.returncode})"
        )

        if not combined_output:
            return f"Command {status} with no output."

        return f"Command {status}:\n{combined_output}"
