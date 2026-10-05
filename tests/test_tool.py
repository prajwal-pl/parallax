import pytest
from typing import Literal, Optional, Union
from core.tool import tool, extract_schema, parse_docstring, resolve_json_type


def test_tool_decorator():
    @tool
    def sample_func(x: int) -> int:
        """Sample function."""
        return x + 1

    assert getattr(sample_func, "is_tool", False) is True


def test_parse_docstring_empty():
    main, params = parse_docstring(None)
    assert main == ""
    assert params == {}

    main, params = parse_docstring("")
    assert main == ""
    assert params == {}


def test_parse_docstring_google_style():
    doc = """
    Replace one exact occurrence of old with new.

    Args:
        path (str): Path to target file.
        old: The exact string to find.
        new: The replacement string
            which spans across multiple lines.
    Returns:
        Confirmation message.
    """
    main, params = parse_docstring(doc)
    assert main == "Replace one exact occurrence of old with new."
    assert params["path"] == "Path to target file."
    assert params["old"] == "The exact string to find."
    assert params["new"] == "The replacement string which spans across multiple lines."


def test_resolve_json_type_primitives():
    assert resolve_json_type(str) == {"type": "string"}
    assert resolve_json_type(int) == {"type": "integer"}
    assert resolve_json_type(float) == {"type": "number"}
    assert resolve_json_type(bool) == {"type": "boolean"}


def test_resolve_json_type_collections():
    assert resolve_json_type(list[str]) == {"type": "array", "items": {"type": "string"}}
    assert resolve_json_type(list[int]) == {"type": "array", "items": {"type": "integer"}}
    assert resolve_json_type(dict) == {"type": "object"}


def test_resolve_json_type_union_and_optional():
    assert resolve_json_type(str | None) == {"type": "string"}
    assert resolve_json_type(Optional[int]) == {"type": "integer"}
    union_res = resolve_json_type(Union[int, str])
    assert "anyOf" in union_res
    assert {"type": "integer"} in union_res["anyOf"]
    assert {"type": "string"} in union_res["anyOf"]


def test_resolve_json_type_literal():
    lit = Literal["view", "create", "str_replace"]
    res = resolve_json_type(lit)
    assert res["type"] == "string"
    assert res["enum"] == ["view", "create", "str_replace"]


def test_extract_schema_comprehensive():
    @tool
    def complex_tool(
        path: str,
        command: Literal["read", "write"],
        lines: list[int] | None = None,
        retries: int = 3,
    ) -> str:
        """
        Execute an operation on a file.

        Args:
            path: Target file path.
            command: Action to perform.
            lines: Optional line numbers to target.
            retries: Number of retry attempts.
        """
        return f"{command} {path}"

    schema = extract_schema(complex_tool)
    as_dict = schema.to_dict()

    assert as_dict["type"] == "function"
    fn = as_dict["function"]
    assert fn["name"] == "complex_tool"
    assert fn["description"] == "Execute an operation on a file."
    
    props = fn["parameters"]["properties"]
    assert props["path"]["type"] == "string"
    assert props["path"]["description"] == "Target file path."

    assert props["command"]["type"] == "string"
    assert props["command"]["enum"] == ["read", "write"]
    assert props["command"]["description"] == "Action to perform."

    assert props["lines"]["type"] == "array"
    assert props["lines"]["items"] == {"type": "integer"}
    assert props["lines"]["description"] == "Optional line numbers to target."

    assert props["retries"]["type"] == "integer"
    assert props["retries"]["default"] == 3
    assert props["retries"]["description"] == "Number of retry attempts."

    # path and command have no default, so they are required
    assert "path" in fn["parameters"]["required"]
    assert "command" in fn["parameters"]["required"]
    # lines and retries have defaults, so they are not required
    assert "lines" not in fn["parameters"]["required"]
    assert "retries" not in fn["parameters"]["required"]


def test_extract_schema_bound_method():
    class DummyService:
        def __init__(self, prefix: str):
            self.prefix = prefix

        @tool
        def process(self, text: str) -> str:
            """
            Process incoming text.

            Args:
                text: Content to process.
            """
            return f"{self.prefix}: {text}"

    service = DummyService(prefix="TEST")
    schema = extract_schema(service.process)
    fn = schema.to_dict()["function"]

    # 'self' must NOT be in properties or required
    assert "self" not in fn["parameters"]["properties"]
    assert "self" not in fn["parameters"]["required"]
    assert "text" in fn["parameters"]["properties"]
