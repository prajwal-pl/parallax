from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from parallax.cli.repl import (
    FALLBACK_GROQ_MODELS,
    FALLBACK_OPENROUTER_MODELS,
    InteractiveRepl,
    SlashCommandCompleter,
    _MODEL_CACHE,
    get_provider_models,
)
from parallax.core.agent import Agent
from parallax.core.llm import (
    GroqClient,
    OpenRouterClient,
    fetch_groq_models,
    fetch_openrouter_models,
)
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console


@pytest.mark.asyncio
async def test_fetch_groq_models_success():
    mock_client = MagicMock()
    mock_m1 = MagicMock()
    mock_m1.id = "llama-3.3-70b-versatile"
    mock_m2 = MagicMock()
    mock_m2.id = "qwen/qwen3.8-27b"
    mock_listing = MagicMock()
    mock_listing.data = [mock_m1, mock_m2]
    mock_client.models.list.return_value = mock_listing

    models = await fetch_groq_models(client=mock_client)
    assert "llama-3.3-70b-versatile" in models
    assert "qwen/qwen3.8-27b" in models
    mock_client.models.list.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_groq_models_with_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import parallax.core.llm as llm_module

    original_groq = llm_module.groq
    try:
        mock_groq_class = MagicMock()
        mock_instance = MagicMock()
        mock_m = MagicMock()
        mock_m.id = "openai/gpt-oss-20b"
        mock_listing = MagicMock()
        mock_listing.data = [mock_m]
        mock_instance.models.list.return_value = mock_listing
        mock_groq_class.return_value = mock_instance

        class FakeGroqModule:
            Groq = mock_groq_class

        llm_module.groq = FakeGroqModule
        models = await fetch_groq_models(api_key="gsk_custom_key")
        assert models == ["openai/gpt-oss-20b"]
        mock_groq_class.assert_called_once_with(api_key="gsk_custom_key")
    finally:
        llm_module.groq = original_groq


@pytest.mark.asyncio
async def test_fetch_openrouter_models_success():
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.json.return_value = {
        "data": [
            {"id": "meta-llama/llama-3.3-70b-instruct"},
            {"id": "google/gemini-2.0-flash"},
        ]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=fake_response)) as mock_get:
        models = await fetch_openrouter_models(api_key="sk-or-test123")
        assert models == ["meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash"]
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://openrouter.ai/api/v1/models"
        assert kwargs["headers"] == {"Authorization": "Bearer sk-or-test123"}


@pytest.mark.asyncio
async def test_fetch_openrouter_models_error():
    with patch("httpx.AsyncClient.get", new=AsyncMock(side_effect=httpx.ConnectError("Connection failed"))):
        with pytest.raises(httpx.ConnectError):
            await fetch_openrouter_models()


@pytest.mark.asyncio
async def test_client_list_models_methods():
    groq_client = GroqClient(api_key="test")
    with patch("parallax.core.llm.fetch_groq_models", new=AsyncMock(return_value=["model-a"])):
        models = await groq_client.list_models()
        assert models == ["model-a"]

    openrouter_client = OpenRouterClient(api_key="test")
    with patch("parallax.core.llm.fetch_openrouter_models", new=AsyncMock(return_value=["model-b"])):
        models = await openrouter_client.list_models()
        assert models == ["model-b"]


@pytest.mark.asyncio
async def test_get_provider_models_caching_and_refresh():
    _MODEL_CACHE.clear()

    # 1. First fetch groq models (mocked)
    with patch("parallax.cli.repl.fetch_groq_models", new=AsyncMock(return_value=["groq-1", "groq-2"])) as mock_groq:
        res1 = await get_provider_models("GroqClient")
        assert "groq-1" in res1 and "groq-2" in res1
        assert "llama-3.3-70b-versatile" in res1
        assert mock_groq.call_count == 1

        # Second fetch should hit cache without calling fetcher again
        res2 = await get_provider_models("GroqClient")
        assert res2 == res1
        assert mock_groq.call_count == 1

        # Force refresh should bypass cache
        res3 = await get_provider_models("GroqClient", force_refresh=True)
        assert res3 == res1
        assert mock_groq.call_count == 2

    # 2. OpenRouter fallback on failure
    _MODEL_CACHE.clear()
    with patch("parallax.cli.repl.fetch_openrouter_models", new=AsyncMock(side_effect=Exception("Network error"))):
        res_fallback = await get_provider_models("OpenRouterClient")
        assert res_fallback == FALLBACK_OPENROUTER_MODELS

    # 3. Groq fallback on failure
    _MODEL_CACHE.clear()
    with patch("parallax.cli.repl.fetch_groq_models", new=AsyncMock(side_effect=Exception("API error"))):
        res_groq_fallback = await get_provider_models("GroqClient")
        assert res_groq_fallback == FALLBACK_GROQ_MODELS


@pytest.mark.asyncio
async def test_repl_prefetch_and_autocomplete():
    _MODEL_CACHE.clear()
    agent = Agent(model="liquid/lfm-2.5-2.6b:free", system_prompt="s")
    repl = InteractiveRepl(agent=agent, console=Console(record=True))

    with patch("parallax.cli.repl.fetch_openrouter_models", new=AsyncMock(return_value=["custom/test-dynamic-model"])):
        await repl._prefetch_models()
        assert "custom/test-dynamic-model" in _MODEL_CACHE["OpenRouterClient"]

    # Verify completer provides the dynamically fetched model
    completer = SlashCommandCompleter(get_agent=lambda: agent)
    doc = Document(text="/model custom/", cursor_position=14)
    completions = [c.text for c in completer.get_completions(doc, CompleteEvent(text_inserted=False))]
    assert "custom/test-dynamic-model" in completions


@pytest.mark.asyncio
async def test_repl_model_provider_filtering():
    from parallax.cli.repl import ModelOption

    agent = Agent(model="liquid/lfm-2.5-2.6b:free", system_prompt="s")
    repl = InteractiveRepl(agent=agent, console=Console(record=True))

    # Test /model groq calls picker with initial_provider_filter="Groq"
    with patch.object(
        repl, "_pick_model", new=AsyncMock(return_value=ModelOption(id="llama-3.3-70b-versatile", provider="Groq"))
    ) as mock_pick:
        await repl._handle_command("/model groq")
        assert mock_pick.call_args.kwargs["initial_provider_filter"] == "Groq"
        assert agent.model == "llama-3.3-70b-versatile"
        assert agent.provider_name == "Groq"

    # Test /model openrouter calls picker with initial_provider_filter="OpenRouter"
    with patch.object(
        repl, "_pick_model", new=AsyncMock(return_value=ModelOption(id="openai/gpt-4o-mini", provider="OpenRouter"))
    ) as mock_pick:
        await repl._handle_command("/model openrouter")
        assert mock_pick.call_args.kwargs["initial_provider_filter"] == "OpenRouter"
        assert agent.model == "openai/gpt-4o-mini"
        assert agent.provider_name == "OpenRouter"
