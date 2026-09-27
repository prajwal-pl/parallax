from dataclasses import dataclass, field, asdict
import json

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
    content: str
    iterations: int
    messages: list[Message]

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    def to_message_dict(self) -> dict:
        if self.tool_calls:
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    },
                } for tc in self.tool_calls]
            }
        return {
            "role": "assistant",
            "content": self.content
        }

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