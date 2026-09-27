from dataclasses import dataclass, field, asdict

@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list | None = None

@dataclass
class AgentEvent:
    pass

@dataclass
class AgentResult:
    pass

@dataclass
class ToolCall:
    pass

@dataclass
class ToolParameters:
    type: str = "object"
    properties: dict = field(default_factory=dict)
    required: list = field(default_factory=list)

@dataclass
class ToolFunction:
    name: str
    description: str
    parameters: ToolParameters = field(default_factory=ToolParameters)

@dataclass
class ToolSchema:
    type: str = "function"
    function: ToolFunction | None = None

    def to_dict(self) -> dict:
        return asdict(self)