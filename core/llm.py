# import ast
import json
import os

from dotenv import load_dotenv
from openrouter import OpenRouter

from core.types import LLMResponse, ToolCall

load_dotenv()

openrouter = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))


# def parse_arguments(raw: str) -> dict:
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         try:
#             return ast.literal_eval(raw)
#         except Exception:
#             return {"raw": raw}


def parse_tool_arguments(raw: str | None) -> tuple[dict, str | None]:
    """
    Parse raw JSON tool arguments from the model.
    Returns (arguments, error). On failure arguments is {} and error describes why,
    so the model can be told and retry rather than crashing the run.
    """
    if raw is None or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON in tool arguments ({e}). Raw: {raw[:200]}"
    if not isinstance(parsed, dict):
        return {}, f"Tool arguments must be a JSON object, got {type(parsed).__name__}."
    return parsed, None


async def call_llm(model: str, messages: list, tools: list) -> LLMResponse:
    """
    Call the LLM with the given model, messages, and tools.
    """

    serialized = [m.to_dict() for m in messages]

    chat = await openrouter.chat.send_async(
        model=model, messages=serialized, tools=tools, tool_choice="auto"
    )

    response = chat.choices[0].message

    chat_response = response.content
    calls = response.tool_calls

    tool_calls: list[ToolCall] | None = None
    if calls:
        tool_calls = []
        for tc in calls:
            arguments, error = parse_tool_arguments(tc.function.arguments)
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                    error=error,
                )
            )

    return LLMResponse(
        content=chat_response if isinstance(chat_response, str) else None,
        tool_calls=tool_calls,
    )
