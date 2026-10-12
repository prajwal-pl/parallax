from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock

from parallax.core.agent import Agent
from parallax.core.llm import GroqClient, OpenRouterClient
from parallax.core.types import LLMResponse, Message, Usage


def test_agent_selects_openrouter_by_default(monkeypatch):
    monkeypatch.delenv("PARALLAX_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = Agent(model="openai/gpt-4o", system_prompt="s")
    assert isinstance(agent.llm, OpenRouterClient)


def test_agent_selects_groq_via_env(monkeypatch):
    monkeypatch.setenv("PARALLAX_PROVIDER", "groq")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from unittest.mock import patch, MagicMock
    with patch("parallax.core.agent.GroqClient") as MockGroq:
        mock_instance = MagicMock()
        MockGroq.return_value = mock_instance
        agent = Agent(model="any-model", system_prompt="s")
        assert MockGroq.called
        assert agent.llm is mock_instance


def test_agent_selects_groq_via_model_prefix(monkeypatch):
    monkeypatch.delenv("PARALLAX_PROVIDER", raising=False)
    # Avoid actual Groq import by mocking GroqClient init
    from unittest.mock import patch
    with patch("parallax.core.agent.GroqClient") as MockGroq:
        mock_instance = MagicMock()
        MockGroq.return_value = mock_instance
        agent = Agent(model="groq/llama-3.1-70b-versatile", system_prompt="s")
        assert MockGroq.called
        assert agent.llm is mock_instance


def test_agent_selects_groq_via_recognized_model(monkeypatch):
    monkeypatch.delenv("PARALLAX_PROVIDER", raising=False)
    from unittest.mock import patch
    with patch.object(GroqClient, "__init__", return_value=None) as mock_init:
        agent = Agent(model="openai/gpt-oss-20b", system_prompt="s")
        assert mock_init.called
        assert isinstance(agent.llm, GroqClient)
        assert agent.provider_name == "Groq"


def test_agent_set_model_switches_provider(monkeypatch):
    monkeypatch.delenv("PARALLAX_PROVIDER", raising=False)
    from unittest.mock import patch
    with patch.object(GroqClient, "__init__", return_value=None), patch.object(OpenRouterClient, "__init__", return_value=None):
        agent = Agent(model="custom/unknown-model", system_prompt="s")
        assert isinstance(agent.llm, OpenRouterClient)

        agent.set_model("openai/gpt-oss-120b")
        assert isinstance(agent.llm, GroqClient)
        assert agent.model == "openai/gpt-oss-120b"
        assert agent.provider_name == "Groq"


@pytest.mark.asyncio
async def test_groq_client_parses_response():
    # Mock groq SDK
    mock_message = MagicMock()
    mock_message.content = "hello"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_chat = MagicMock()
    mock_chat.choices = [mock_choice]
    mock_chat.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_completions = MagicMock()
    mock_completions.create = MagicMock(return_value=mock_chat)

    mock_chat_attr = MagicMock()
    mock_chat_attr.completions = mock_completions

    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat_attr

    # Patch groq.Groq to return our mock
    import parallax.core.llm as llm_module
    original_groq = llm_module.groq
    try:
        class FakeGroqModule:
            Groq = lambda api_key=None: mock_groq_instance
        llm_module.groq = FakeGroqModule
        client = GroqClient(api_key="test")
        resp = await client.generate("llama-3.1-8b-instant", [Message(role="user", content="hi")], [])
        assert resp.content == "hello"
        assert resp.usage == Usage(input_tokens=10, output_tokens=5)
        assert resp.tool_calls is None
    finally:
        llm_module.groq = original_groq
