from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    def to_dict(self) -> dict:
        # Tool result message (sent back after executing a tool)
        if self.tool_call_id:
            return {
                "role": self.role,
                "content": self.content,
                "tool_call_id": self.tool_call_id,
            }
        # Assistant message that requested tool calls
        if self.tool_calls:
            tool_calls = self.tool_calls  # local var so Pylance can narrow
            return {
                "role": self.role,
                "content": self.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        # Plain message (system, user, or final assistant response)
        return {"role": self.role, "content": self.content}


class EventType(str, Enum):
    RUN_STARTED = ("run_started",)
    MODEL_STARTED = ("model_started",)
    TOOL_STARTED = ("tool_started",)
    TOOL_COMPLETED = ("tool_completed",)
    MODEL_COMPLETED = ("model_completed",)
    RUN_COMPLETED = ("run_completed",)
    RUN_FAILED = "run_failed"
    TOOL_FAILED = "tool_failed"


@dataclass
class AgentEvent:
    type: EventType
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


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

    def to_message(self) -> Message:
        if self.tool_calls:
            return Message(
                role="assistant", content=self.content or "", tool_calls=self.tool_calls
            )
        else:
            return Message(
                role="assistant",
                content=self.content or "",
            )

    def to_message_dict(self) -> dict:
        if self.tool_calls:
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in self.tool_calls
                ],
            }
        return {"role": "assistant", "content": self.content}


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
