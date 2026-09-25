from openrouter import OpenRouter
import os
from dotenv import load_dotenv

load_dotenv()

openrouter = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

def call_llm(model:str, messages:list, tools:list):
    """
    Call the LLM with the given model, messages, and tools.
    """

    chat = openrouter.chat.send(model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto")

    response = chat.choices[0].message
    return response