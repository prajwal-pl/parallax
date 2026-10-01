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

    return LLMResponse(
        content=chat_response if isinstance(chat_response, str) else None,
        tool_calls=[
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in calls
        ]
        if calls
        else None,
    )
