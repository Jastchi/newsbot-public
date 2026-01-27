"""Tests for utility functions and database operations."""

import sys
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from utilities import models as config_models
from utilities import setup_django
from utilities.django_models import Article as DjangoArticle


def test_article_repr_shows_title_and_source():
    setup_django()
    article = DjangoArticle(
        config_file="test",
        title="Hello World",
        content="",
        source="Source",
        url="u",
    )
    assert "test" in str(article)
    assert "Hello World" in str(article)
    assert "Source" in str(article)


@pytest.mark.django_db
def test_django_article_save_and_query():
    """Test that Django Article can be saved and queried."""
    # Django is automatically set up when django_models is imported

    # Create an article
    article = DjangoArticle.objects.create(
        config_file="test",
        title="Test Article",
        content="Test content",
        source="Test Source",
        url="http://test.com/article1",
    )

    # Verify it was saved
    articles = DjangoArticle.objects.all()
    assert len(articles) == 1
    assert articles[0].title == "Test Article"
    assert articles[0].id == article.id


class TestLLMProvider:
    """Test cases for LLM provider abstraction."""

    def test_get_llm_provider_returns_ollama_by_default(self):
        """Test that Ollama provider is returned by default."""
        from newsbot.llm_provider import OllamaProvider, get_llm_provider

        config = config_models.ConfigModel()

        with patch.object(
            OllamaProvider, "_initialize_client", return_value=None
        ):
            with patch.object(
                OllamaProvider, "_ensure_model_available", return_value=None
            ):
                provider = get_llm_provider(config)
                assert isinstance(provider, OllamaProvider)

    def test_get_llm_provider_returns_ollama_explicitly(self):
        """Test that Ollama provider is returned when specified."""
        from newsbot.llm_provider import OllamaProvider, get_llm_provider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(llm=LLMConfigModel(provider="ollama"))

        with patch.object(
            OllamaProvider, "_initialize_client", return_value=None
        ):
            with patch.object(
                OllamaProvider, "_ensure_model_available", return_value=None
            ):
                provider = get_llm_provider(config)
                assert isinstance(provider, OllamaProvider)

    def test_get_llm_provider_returns_gemini(self, monkeypatch):
        """Test that Gemini provider is returned when specified."""
        from newsbot.llm_provider import GeminiProvider, get_llm_provider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )

        # Mock the Gemini SDK and API key
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

        mock_genai = MagicMock()
        mock_genai.Client.return_value = MagicMock()
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = get_llm_provider(config)
            assert isinstance(provider, GeminiProvider)

    def test_ollama_provider_generate(self):
        """Test OllamaProvider.generate method."""
        from newsbot.llm_provider import OllamaProvider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(
            llm=LLMConfigModel(
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0.7,
                max_tokens=2000,
            ),
        )

        with patch.object(
            OllamaProvider, "_initialize_client", return_value=None
        ):
            with patch.object(
                OllamaProvider, "_ensure_model_available", return_value=None
            ):
                with patch("ollama.generate") as mock_gen:
                    mock_gen.return_value = {"response": "Test response"}

                    provider = OllamaProvider(config)
                    result = provider.generate(
                        "Test prompt",
                        {"temperature": 0.5, "num_predict": 100},
                    )

                    assert result == "Test response"
                    mock_gen.assert_called_once()

    def test_ollama_provider_chat(self):
        """Test OllamaProvider.chat method."""
        from newsbot.llm_provider import OllamaProvider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(
            llm=LLMConfigModel(
                model="llama2",
                base_url="http://localhost:11434",
            ),
        )

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "Chat response"}
        }
        mock_client.list.return_value = {"models": [{"model": "llama2"}]}

        with patch.object(
            OllamaProvider, "_initialize_client", return_value=None
        ):
            with patch.object(
                OllamaProvider, "_ensure_model_available", return_value=None
            ):
                provider = OllamaProvider(config)
                provider.client = mock_client
                result = provider.chat(
                    [{"role": "user", "content": "Hello"}],
                    {"temperature": 0.5},
                )

                assert result == "Chat response"

    def test_gemini_provider_requires_api_key(self, monkeypatch):
        """Test that GeminiProvider raises error without API key."""
        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )

        # Remove API key if set
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_genai = MagicMock()
        mock_types = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiProvider(config)

    def test_get_llm_provider_falls_back_for_unknown(self):
        """Test that unknown provider falls back to Ollama."""
        from newsbot.llm_provider import OllamaProvider, get_llm_provider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(
            llm=LLMConfigModel(provider="unknown_provider"),
        )

        with patch.object(
            OllamaProvider, "_initialize_client", return_value=None
        ):
            with patch.object(
                OllamaProvider, "_ensure_model_available", return_value=None
            ):
                provider = get_llm_provider(config)
                assert isinstance(provider, OllamaProvider)

    def test_ollama_provider_chat_json(self):
        """Test OllamaProvider.chat_json method."""
        from newsbot.llm_provider import OllamaProvider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(
            llm=LLMConfigModel(
                model="llama2",
                base_url="http://localhost:11434",
            ),
        )

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": '{"result": "success", "value": 42}'}
        }

        with patch.object(
            OllamaProvider, "_initialize_client", return_value=None
        ):
            with patch.object(
                OllamaProvider, "_ensure_model_available", return_value=None
            ):
                provider = OllamaProvider(config)
                provider.client = mock_client
                result = provider.chat_json(
                    [{"role": "user", "content": "Get JSON"}],
                    {"temperature": 0.5},
                    {"type": "object"},
                )

                assert result == {"result": "success", "value": 42}
                mock_client.chat.assert_called_once()

    def test_ollama_initialize_client_starts_service_on_failure(self):
        """Test _initialize_client starts service when Ollama not running."""
        from newsbot.llm_provider import OllamaProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(model="llama2"),
        )

        mock_client = MagicMock()
        # First call fails, second succeeds after service starts
        mock_client.list.side_effect = [
            Exception("Not running"),
            {"models": []},
        ]

        with patch("ollama.Client", return_value=mock_client):
            with patch.object(
                OllamaProvider, "_start_service", return_value=True
            ) as mock_start:
                with patch.object(
                    OllamaProvider,
                    "_ensure_model_available",
                    return_value=None,
                ):
                    provider = OllamaProvider(config)
                    mock_start.assert_called_once()

    def test_ollama_initialize_client_exits_on_service_failure(self):
        """Test _initialize_client exits when service cannot start."""
        from newsbot.llm_provider import OllamaProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(model="llama2"),
        )

        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Not running")

        with patch("ollama.Client", return_value=mock_client):
            with patch.object(
                OllamaProvider, "_start_service", return_value=False
            ):
                with pytest.raises(SystemExit):
                    OllamaProvider(config)

    def test_ollama_start_service_success(self):
        """Test _start_service successfully starts Ollama."""
        from newsbot.llm_provider import OllamaProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(model="llama2"),
        )

        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}

        with patch("ollama.Client", return_value=mock_client):
            with patch("subprocess.Popen"):
                with patch("time.sleep"):
                    with patch.object(
                        OllamaProvider,
                        "_ensure_model_available",
                        return_value=None,
                    ):
                        provider = OllamaProvider(config)
                        provider.client = mock_client
                        result = provider._start_service()
                        assert result is True

    def test_ollama_start_service_failure(self):
        """Test _start_service returns False on failure."""
        from newsbot.llm_provider import OllamaProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(model="llama2"),
        )

        with patch.object(
            OllamaProvider, "_initialize_client", return_value=None
        ):
            with patch.object(
                OllamaProvider, "_ensure_model_available", return_value=None
            ):
                provider = OllamaProvider(config)

        with patch(
            "subprocess.Popen", side_effect=Exception("Failed to start")
        ):
            result = provider._start_service()
            assert result is False

    def test_ollama_ensure_model_available_pulls_missing_model(self):
        """Test _ensure_model_available pulls a missing model."""
        from newsbot.llm_provider import OllamaProvider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(llm=LLMConfigModel(model="new-model"))

        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"model": "other-model"}]}

        def init_with_mock_client(self_provider):
            self_provider.client = mock_client

        with patch.object(
            OllamaProvider, "_initialize_client", init_with_mock_client
        ):
            with patch("ollama.pull") as mock_pull:
                provider = OllamaProvider(config)
                mock_pull.assert_called_once_with("new-model")

    def test_ollama_ensure_model_available_skips_existing_model(self):
        """Test _ensure_model_available skips existing model."""
        from newsbot.llm_provider import OllamaProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(model="llama2"),
        )

        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"model": "llama2"}]}

        def init_with_mock_client(self_provider):
            self_provider.client = mock_client

        with patch.object(
            OllamaProvider, "_initialize_client", init_with_mock_client
        ):
            with patch("ollama.pull") as mock_pull:
                OllamaProvider(config)
                mock_pull.assert_not_called()

    def test_ollama_ensure_model_available_exits_on_error(self):
        """Test _ensure_model_available exits on pull failure."""
        from newsbot.llm_provider import OllamaProvider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(llm=LLMConfigModel(model="bad-model"))

        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}

        def init_with_mock_client(self_provider):
            self_provider.client = mock_client

        with patch.object(
            OllamaProvider, "_initialize_client", init_with_mock_client
        ):
            with patch("ollama.pull", side_effect=Exception("Pull failed")):
                with pytest.raises(SystemExit):
                    OllamaProvider(config)

    def test_gemini_provider_generate(self, monkeypatch):
        """Test GeminiProvider.generate method."""
        from newsbot.llm_provider import GeminiProvider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(
            llm=LLMConfigModel(provider="gemini", model="gemini-pro"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.text = "  Generated text  "
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = GeminiProvider(config)
            provider.client = mock_client
            result = provider.generate(
                "Test prompt", {"temperature": 0.5, "num_predict": 100}
            )

            assert result == "Generated text"
            mock_client.models.generate_content.assert_called_once()

    def test_gemini_provider_generate_empty_response(self, monkeypatch):
        """Test GeminiProvider.generate handles empty response."""
        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.text = None
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "SAFETY"
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = GeminiProvider(config)
            provider.client = mock_client
            result = provider.generate("Test prompt", {})

            assert result == ""

    def test_gemini_provider_chat(self, monkeypatch):
        """Test GeminiProvider.chat method."""
        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.text = "  Chat response  "

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = GeminiProvider(config)
            provider.client = mock_client
            result = provider.chat(
                [{"role": "user", "content": "Hello"}], {"temperature": 0.5}
            )

            assert result == "Chat response"

    def test_gemini_provider_chat_empty_response(self, monkeypatch):
        """Test GeminiProvider.chat handles empty response."""
        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = GeminiProvider(config)
            provider.client = mock_client
            result = provider.chat([{"role": "user", "content": "Hello"}], {})

            assert result == ""

    def test_gemini_provider_chat_json(self, monkeypatch):
        """Test GeminiProvider.chat_json method."""
        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.text = '{"status": "ok", "count": 5}'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = GeminiProvider(config)
            provider.client = mock_client
            result = provider.chat_json(
                [{"role": "user", "content": "Get JSON"}],
                {"temperature": 0.3},
                {"type": "object"},
            )

            assert result == {"status": "ok", "count": 5}

    def test_gemini_provider_chat_json_empty_response(self, monkeypatch):
        """Test GeminiProvider.chat_json handles empty response."""
        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = GeminiProvider(config)
            provider.client = mock_client
            result = provider.chat_json(
                [{"role": "user", "content": "Get JSON"}], {}, {}
            )

            assert result == {}

    def test_gemini_provider_convert_messages(self, monkeypatch):
        """Test GeminiProvider._convert_messages method."""
        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_genai = MagicMock()
        mock_genai.Client.return_value = MagicMock()

        # Track Content calls to verify role conversion
        content_calls = []

        def track_content_call(role, parts):
            content_calls.append({"role": role, "parts": parts})
            return MagicMock()

        mock_types = MagicMock()
        mock_types.Content.side_effect = track_content_call
        mock_types.Part.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            provider = GeminiProvider(config)
            # The _types attribute was set during init, update it to use our mock
            provider._types = mock_types

            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            result = provider._convert_messages(messages)

            assert len(result) == 3
            # Verify role conversions
            assert content_calls[0]["role"] == "user"  # system -> user
            assert content_calls[1]["role"] == "user"  # user stays user
            assert content_calls[2]["role"] == "model"  # assistant -> model

    @pytest.mark.use_real_ollama
    def test_ollama_provider_real_call(self, monkeypatch):
        """Test OllamaProvider with real ollama calls when enabled and not in CI.

        This test will only make real ollama API calls if:
        - ENABLE_OLLAMA environment variable is set to 'true'
        - Not running in CI (or MOCK_OLLAMA is set to 'false')
        - Ollama service is available and running

        To run this test with real ollama:
        ENABLE_OLLAMA=true pytest tests/core_tests/test_utils_and_database.py::TestLLMProvider::test_ollama_provider_real_call
        """
        import os

        is_github_action = os.getenv("GITHUB_ACTIONS") == "true"
        mock_ollama = os.getenv("MOCK_OLLAMA")
        enable_ollama = os.getenv("ENABLE_OLLAMA", "false").lower() == "true"

        # Skip if conditions not met
        if not enable_ollama:
            pytest.skip(
                "ENABLE_OLLAMA not set to 'true'. Skipping real ollama test."
            )

        if is_github_action and mock_ollama != "false":
            pytest.skip(
                "Running in CI with MOCK_OLLAMA not set to 'false'. "
                "Skipping real ollama test."
            )

        from newsbot.llm_provider import OllamaProvider
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(
            llm=LLMConfigModel(
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0.7,
                max_tokens=100,  # Keep it short for testing
            ),
        )

        try:
            # This will make a real call to ollama if it's running
            provider = OllamaProvider(config)

            # Test generate method with real ollama
            result = provider.generate(
                "Say 'Hello, this is a test' and nothing else.",
                {"temperature": 0.1, "num_predict": 50},
            )

            # Verify we got a response
            assert result is not None
            assert len(result) > 0
            assert isinstance(result, str)

            # Test chat method with real ollama
            chat_result = provider.chat(
                [
                    {
                        "role": "user",
                        "content": "Say 'Test successful' and nothing else.",
                    }
                ],
                {"temperature": 0.1, "num_predict": 50},
            )

            assert chat_result is not None
            assert len(chat_result) > 0
            assert isinstance(chat_result, str)

        except Exception as e:
            # If ollama is not available, skip the test
            pytest.skip(f"Ollama not available or not running: {e}")


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_setup_logging_with_config_key(self, tmp_path, monkeypatch):
        """Test setup_logging derives log filename from config key."""
        import logging

        from newsbot.utils import setup_logging

        # Change to tmp_path to avoid polluting the project directory
        monkeypatch.chdir(tmp_path)

        # Clear existing handlers to test fresh setup
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()

        from utilities.models import ConfigModel, LoggingConfigModel

        try:
            config = ConfigModel(
                logging=LoggingConfigModel(level="DEBUG"),
            )
            setup_logging(config, [], config_key="technology")

            # Verify logs directory was created
            assert (tmp_path / "logs").exists()
            # Verify log file path is derived from config key
            assert (tmp_path / "logs" / "technology.log").exists()
        finally:
            # Restore original handlers
            root_logger.handlers = original_handlers

    def test_setup_logging_without_config_key(self, tmp_path, monkeypatch):
        """Test setup_logging uses default log filename when no config key."""
        import logging

        from newsbot.utils import setup_logging

        monkeypatch.chdir(tmp_path)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()

        try:
            config = config_models.ConfigModel()
            setup_logging(config, [])

            assert (tmp_path / "logs").exists()
            # Default log file should be newsbot.log
            assert (tmp_path / "logs" / "newsbot.log").exists()
        finally:
            root_logger.handlers = original_handlers

    def test_setup_logging_custom_format(self, tmp_path, monkeypatch):
        """Test setup_logging respects custom log format."""
        import logging

        from newsbot.utils import setup_logging

        monkeypatch.chdir(tmp_path)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()

        try:
            from utilities.models import ConfigModel, LoggingConfigModel

            custom_format = "%(levelname)s - %(message)s"
            config = ConfigModel(
                logging=LoggingConfigModel(level="INFO", format=custom_format),
            )
            setup_logging(config, [], config_key="test")

            # Verify logs directory was created
            assert (tmp_path / "logs").exists()
        finally:
            root_logger.handlers = original_handlers

    def test_setup_logging_with_error_handlers(self, tmp_path, monkeypatch):
        """Test setup_logging includes custom error handlers."""
        import logging

        from newsbot.utils import setup_logging

        monkeypatch.chdir(tmp_path)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()

        try:
            # Use a real NullHandler instead of mock to avoid formatter issues
            custom_handler = logging.NullHandler()
            config = config_models.ConfigModel()

            setup_logging(config, [custom_handler], config_key="test")

            # Verify handler was added to root logger
            assert custom_handler in logging.getLogger().handlers
        finally:
            root_logger.handlers = original_handlers


class TestValidateEnvironment:
    """Test cases for validate_environment function."""

    def test_validate_environment_passes_with_gemini_api_key(
        self,
        monkeypatch,
    ):
        """Test that validation passes when GEMINI_API_KEY is set for gemini."""
        from newsbot.utils import validate_environment

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

        # Should not raise or exit
        validate_environment(config)

    def test_validate_environment_passes_with_ollama_provider(
        self,
        monkeypatch,
    ):
        """Test that validation passes for ollama provider (no API key needed)."""
        from newsbot.utils import validate_environment
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(llm=LLMConfigModel(provider="ollama"))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Should not raise or exit
        validate_environment(config)

    def test_validate_environment_passes_with_default_provider(
        self,
        monkeypatch,
    ):
        """Test that validation passes with default provider (ollama)."""
        from newsbot.utils import validate_environment

        config = config_models.ConfigModel()
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Should not raise or exit
        validate_environment(config)

    def test_validate_environment_exits_without_gemini_api_key(
        self,
        monkeypatch,
        capsys,
    ):
        """Test that validation exits when GEMINI_API_KEY is missing for gemini."""
        from newsbot.utils import validate_environment

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            validate_environment(config)

        # Should exit with code 0 (to prevent watchmedo restart)
        assert exc_info.value.code == 0

        # Check error message is printed
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "GEMINI_API_KEY" in captured.out
        assert "gemini" in captured.out

    def test_validate_environment_logs_error(self, monkeypatch, caplog):
        """Test that validation logs error when environment variable is missing."""
        import logging

        from newsbot.utils import validate_environment

        caplog.set_level(logging.ERROR, logger="newsbot.utils")

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(SystemExit):
            validate_environment(config)

        assert "GEMINI_API_KEY" in caplog.text
        assert "gemini" in caplog.text

    def test_validate_environment_flushes_email_handler(
        self,
        monkeypatch,
    ):
        """Test that email error handler is flushed before exiting."""
        from newsbot.utils import validate_environment

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_email_handler = MagicMock()
        mock_email_handler.flush = MagicMock()

        with pytest.raises(SystemExit):
            validate_environment(config, mock_email_handler)

        # Verify flush was called
        mock_email_handler.flush.assert_called_once()

    def test_validate_environment_handles_email_flush_failure(
        self,
        monkeypatch,
        caplog,
    ):
        """Test that email flush failure doesn't prevent exit."""
        import logging

        from newsbot.utils import validate_environment

        caplog.set_level(logging.WARNING, logger="newsbot.utils")

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_email_handler = MagicMock()
        mock_email_handler.flush = MagicMock(
            side_effect=OSError("Flush failed")
        )

        with pytest.raises(SystemExit) as exc_info:
            validate_environment(config, mock_email_handler)

        # Should still exit
        assert exc_info.value.code == 0

        # Should log warning about flush failure
        assert "Failed to flush email error handler" in caplog.text

    def test_validate_environment_without_email_handler(
        self,
        monkeypatch,
    ):
        """Test that validation works without email error handler."""
        from newsbot.utils import validate_environment

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Should exit even without email handler
        with pytest.raises(SystemExit) as exc_info:
            validate_environment(config, None)

        assert exc_info.value.code == 0

    def test_validate_environment_with_email_handler_no_flush_method(
        self,
        monkeypatch,
    ):
        """Test that validation handles email handler without flush method."""
        from newsbot.utils import validate_environment

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(provider="gemini"),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Handler without flush method
        mock_email_handler = MagicMock()
        del mock_email_handler.flush

        # Should exit without trying to flush
        with pytest.raises(SystemExit) as exc_info:
            validate_environment(config, mock_email_handler)

        assert exc_info.value.code == 0
