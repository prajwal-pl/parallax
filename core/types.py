from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list | None = None



