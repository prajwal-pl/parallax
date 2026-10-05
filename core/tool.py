import inspect
import re
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Union, get_args, get_origin

from core.types import ToolFunction, ToolParameters, ToolSchema


def tool(func: Callable) -> Callable:
    """
    decorator to mark a function as a tool
    """
    setattr(func, "is_tool", True)
    return func


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """
    Extracts the main tool description and individual parameter descriptions
    from standard Google-style or simple docstrings.
    """
    if not doc:
        return "", {}

    lines = doc.strip().splitlines()
    main_desc_lines: list[str] = []
    param_descs: dict[str, str] = {}

    in_args_section = False
    passed_main_section = False
    current_param: str | None = None

    arg_pattern = re.compile(
        r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$"
    )

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "parameters:", "arguments:"):
            in_args_section = True
            passed_main_section = True
            current_param = None
            continue
        elif stripped.lower() in (
            "returns:",
            "raises:",
            "yields:",
            "examples:",
            "note:",
            "notes:",
        ):
            in_args_section = False
            passed_main_section = True
            current_param = None
            continue

        if not passed_main_section:
            main_desc_lines.append(stripped)
        elif in_args_section:
            match = arg_pattern.match(line)
            if match:
                param_name = match.group(1)
                desc = match.group(2).strip()
                param_descs[param_name] = desc
                current_param = param_name
            elif current_param and stripped and (line.startswith(" ") or line.startswith("\t")):
                param_descs[current_param] = (
                    param_descs[current_param] + " " + stripped
                ).strip()

    main_description = "\n".join(main_desc_lines).strip()
    return main_description, param_descs


def resolve_json_type(annotation: Any) -> dict[str, Any]:
    """
    Recursively maps Python type annotations to JSON Schema definitions.
    Supports: str, int, float, bool, list[T], Optional[T], Union, Literal[...].
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # 1. Handle Union and Optional (e.g. str | None, Optional[int], Union[str, int])
    if origin in (Union, types.UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return resolve_json_type(non_none_args[0])
        elif non_none_args:
            return {"anyOf": [resolve_json_type(arg) for arg in non_none_args]}
        return {"type": "null"}

    # 2. Handle Literal choices (e.g. Literal["view", "create"]) -> enum
    if origin is Literal:
        enum_values = list(args)
        val_type = (
            "string"
            if all(isinstance(v, str) for v in enum_values)
            else "integer"
        )
        return {"type": val_type, "enum": enum_values}

    # 3. Handle Arrays/Lists (e.g. list[int], Sequence[str])
    if origin in (list, tuple, set, Sequence) or annotation in (
        list,
        tuple,
        set,
        Sequence,
    ):
        item_type = resolve_json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}

    # 4. Handle Dictionaries
    if origin in (dict, Mapping) or annotation in (dict, Mapping):
        return {"type": "object"}

    # 5. Handle Primitive Types
    primitives = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    if annotation in primitives:
        return {"type": primitives[annotation]}

    return {"type": "string"}


def extract_schema(func: Callable) -> ToolSchema:
    """
    Converts a Python function's signature and docstring into a ToolSchema.
    """
    main_description, param_docs = parse_docstring(func.__doc__)
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        schema_def = resolve_json_type(param.annotation)

        # Inject description from docstring, fallback to param_name
        schema_def["description"] = param_docs.get(param_name, param_name)

        # Add default value if present and primitive
        if param.default is not inspect.Parameter.empty and isinstance(
            param.default, (str, int, float, bool, type(None))
        ):
            schema_def["default"] = param.default

        properties[param_name] = schema_def

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return ToolSchema(
        type="function",
        function=ToolFunction(
            name=func.__name__,
            description=main_description or func.__name__,
            parameters=ToolParameters(
                type="object",
                properties=properties,
                required=required,
            ),
        ),
    )
