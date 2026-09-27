from openrouter import OpenRouter
from openrouter.types import UNSET
from core.types import LLMResponse, ToolCall
import json
import os
from dotenv import load_dotenv

load_dotenv()

openrouter = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

def call_llm(model:str, messages:list, tools:list) -> LLMResponse:
    """
    Call the LLM with the given model, messages, and tools.
    """

    chat = openrouter.chat.send(model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto")

    response = chat.choices[0].message

    chat_response = response.content
    calls = response.tool_calls

    return LLMResponse(
        content=chat_response if isinstance(chat_response, str) else None,
        tool_calls=[ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments)
        ) for tc in calls] if calls else None
    )