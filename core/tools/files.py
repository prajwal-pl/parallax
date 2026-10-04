from pathlib import Path

from core.tool import tool


class FileTools:
    def __init__(self, workspace: str):
        self.workspace = workspace

    def validate_path(self, path: str) -> Path:
        workspace_path = Path(self.workspace).resolve()
        file_path = (Path(self.workspace) / path).resolve()
        if not str(file_path).startswith(str(workspace_path)):
            raise PermissionError(f"Path is outside the workspace: {path}")
        return file_path

    @tool
    async def read_file(self, path: str) -> str:
        """Read file content"""
        try:
            resolved = self.validate_path(path)
            if not resolved.is_file():
                return f"File not found: {path}"
            return resolved.read_text(encoding="utf-8")
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    async def write_file(self, path: str, content: str):
        """Write file content"""
        pass

    @tool
    async def list_files(self):
        """List all available files"""
        pass

    @tool
    async def edit_file(self, path: str, content: str):
        """Edit file content"""
        pass
