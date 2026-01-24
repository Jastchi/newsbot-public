"""Tests for LLM Provider retry logic.

Tests the Gemini retry mechanism with exponential backoff for rate limits
and service unavailable errors.
"""

from unittest.mock import MagicMock, patch

import pytest

from newsbot.llm_provider import (
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_TOO_MANY_REQUESTS,
    GeminiProvider,
    _is_gemini_retryable_error,
)
from utilities import models as config_models


class TestIsGeminiRetryableError:
    """Test cases for _is_gemini_retryable_error detection function."""

    def test_returns_true_for_429_client_error(self):
        """Test that 429 ClientError is correctly identified."""
        from google.genai.errors import ClientError

        error = ClientError(
            HTTP_TOO_MANY_REQUESTS, {"error": {"message": "Rate limit"}}
        )
        assert _is_gemini_retryable_error(error) is True

    def test_returns_true_for_503_server_error(self):
        """Test that 503 ServerError is correctly identified."""
        from google.genai.errors import ServerError

        error = ServerError(
            HTTP_SERVICE_UNAVAILABLE,
            {"error": {"message": "The model is overloaded"}},
        )
        assert _is_gemini_retryable_error(error) is True

    def test_returns_false_for_400_client_error(self):
        """Test that 400 ClientError is not identified as retryable."""
        from google.genai.errors import ClientError

        error = ClientError(400, {"error": {"message": "Bad request"}})
        assert _is_gemini_retryable_error(error) is False

    def test_returns_false_for_401_client_error(self):
        """Test that 401 ClientError is not identified as retryable."""
        from google.genai.errors import ClientError

        error = ClientError(401, {"error": {"message": "Unauthorized"}})
        assert _is_gemini_retryable_error(error) is False

    def test_returns_false_for_403_client_error(self):
        """Test that 403 ClientError is not identified as retryable."""
        from google.genai.errors import ClientError

        error = ClientError(403, {"error": {"message": "Forbidden"}})
        assert _is_gemini_retryable_error(error) is False

    def test_returns_false_for_404_client_error(self):
        """Test that 404 ClientError is not identified as retryable."""
        from google.genai.errors import ClientError

        error = ClientError(404, {"error": {"message": "Not found"}})
        assert _is_gemini_retryable_error(error) is False

    def test_returns_false_for_500_server_error(self):
        """Test that 500 ServerError is not identified as retryable."""
        from google.genai.errors import ServerError

        error = ServerError(500, {"error": {"message": "Internal error"}})
        assert _is_gemini_retryable_error(error) is False

    def test_returns_false_for_generic_exception(self):
        """Test that generic Exception is not identified as retryable."""
        error = Exception("Some error")
        assert _is_gemini_retryable_error(error) is False

    def test_returns_false_for_value_error(self):
        """Test that ValueError is not identified as retryable."""
        error = ValueError("Invalid value")
        assert _is_gemini_retryable_error(error) is False

    def test_returns_false_for_connection_error(self):
        """Test that ConnectionError is not identified as retryable."""
        error = ConnectionError("Connection failed")
        assert _is_gemini_retryable_error(error) is False


class TestGeminiProviderRetry:
    """Test cases for GeminiProvider retry behavior."""

    @pytest.fixture
    def mock_genai(self):
        """Mock the google.genai module."""
        with patch("newsbot.llm_provider.genai") as mock:
            yield mock

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    @patch("newsbot.llm_provider.time.sleep")  # Speed up tests
    def test_generate_retries_on_rate_limit(self, mock_sleep):
        """Test that generate retries on 429 rate limit error."""
        from google.genai.errors import ClientError

        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First call fails with 429, second succeeds
            mock_response = MagicMock()
            mock_response.text = "Success response"
            mock_response.candidates = []

            mock_client.models.generate_content.side_effect = [
                ClientError(429, {"error": {"message": "Rate limit"}}),
                mock_response,
            ]

            provider = GeminiProvider(config)
            result = provider.generate("Test prompt", {})

            assert result == "Success response"
            assert mock_client.models.generate_content.call_count == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    @patch("newsbot.llm_provider.time.sleep")  # Speed up tests
    def test_generate_retries_on_service_unavailable(self, mock_sleep):
        """Test that generate retries on 503 service unavailable error."""
        from google.genai.errors import ServerError

        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First call fails with 503, second succeeds
            mock_response = MagicMock()
            mock_response.text = "Success response"
            mock_response.candidates = []

            mock_client.models.generate_content.side_effect = [
                ServerError(
                    503,
                    {"error": {"message": "The model is overloaded"}},
                ),
                mock_response,
            ]

            provider = GeminiProvider(config)
            result = provider.generate("Test prompt", {})

            assert result == "Success response"
            assert mock_client.models.generate_content.call_count == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    @patch("newsbot.llm_provider.time.sleep")  # Speed up tests
    def test_chat_retries_on_rate_limit(self, mock_sleep):
        """Test that chat retries on 429 rate limit error."""
        from google.genai.errors import ClientError

        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First call fails with 429, second succeeds
            mock_response = MagicMock()
            mock_response.text = "Chat response"
            mock_response.candidates = []

            mock_client.models.generate_content.side_effect = [
                ClientError(429, {"error": {"message": "Rate limit"}}),
                mock_response,
            ]

            provider = GeminiProvider(config)
            result = provider.chat(
                [{"role": "user", "content": "Hello"}],
                {},
            )

            assert result == "Chat response"
            assert mock_client.models.generate_content.call_count == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    @patch("newsbot.llm_provider.time.sleep")  # Speed up tests
    def test_chat_json_retries_on_rate_limit(self, mock_sleep):
        """Test that chat_json retries on 429 rate limit error."""
        from google.genai.errors import ClientError

        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First call fails with 429, second succeeds
            mock_response = MagicMock()
            mock_response.text = '{"result": "success"}'
            mock_response.candidates = []

            mock_client.models.generate_content.side_effect = [
                ClientError(429, {"error": {"message": "Rate limit"}}),
                mock_response,
            ]

            provider = GeminiProvider(config)
            result = provider.chat_json(
                [{"role": "user", "content": "Hello"}],
                {},
                {"type": "object"},
            )

            assert result == {"result": "success"}
            assert mock_client.models.generate_content.call_count == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    def test_does_not_retry_on_400_error(self):
        """Test that non-retryable errors are not retried."""
        from google.genai.errors import ClientError

        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Return 400 error - should not retry
            mock_client.models.generate_content.side_effect = ClientError(
                400,
                {"error": {"message": "Bad request"}},
            )

            provider = GeminiProvider(config)

            with pytest.raises(ClientError) as exc_info:
                provider.generate("Test prompt", {})

            assert exc_info.value.code == 400
            # Should only be called once (no retry)
            assert mock_client.models.generate_content.call_count == 1

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    def test_does_not_retry_on_500_server_error(self):
        """Test that 500 ServerError is not retried."""
        from google.genai.errors import ServerError

        from newsbot.llm_provider import GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Return 500 error - should not retry
            mock_client.models.generate_content.side_effect = ServerError(
                500,
                {"error": {"message": "Internal server error"}},
            )

            provider = GeminiProvider(config)

            with pytest.raises(ServerError) as exc_info:
                provider.generate("Test prompt", {})

            assert exc_info.value.code == 500
            # Should only be called once (no retry)
            assert mock_client.models.generate_content.call_count == 1

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    @patch("newsbot.llm_provider.time.sleep")  # Speed up tests
    def test_exhausts_retries_on_persistent_rate_limit(self, mock_sleep):
        """Test that retries are exhausted on persistent 429 errors."""
        from google.genai.errors import ClientError

        from newsbot.llm_provider import GEMINI_MAX_RETRIES, GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Always return 429 error
            mock_client.models.generate_content.side_effect = ClientError(
                429,
                {"error": {"message": "Rate limit"}},
            )

            provider = GeminiProvider(config)

            with pytest.raises(ClientError) as exc_info:
                provider.generate("Test prompt", {})

            assert exc_info.value.code == 429
            # Should be called GEMINI_MAX_RETRIES times
            assert (
                mock_client.models.generate_content.call_count
                == GEMINI_MAX_RETRIES
            )

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    @patch("newsbot.llm_provider.time.sleep")  # Speed up tests
    def test_exhausts_retries_on_persistent_service_unavailable(
        self, mock_sleep
    ):
        """Test that retries are exhausted on persistent 503 errors."""
        from google.genai.errors import ServerError

        from newsbot.llm_provider import GEMINI_MAX_RETRIES, GeminiProvider

        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                provider="gemini",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
            ),
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Always return 503 error
            mock_client.models.generate_content.side_effect = ServerError(
                503,
                {"error": {"message": "The model is overloaded"}},
            )

            provider = GeminiProvider(config)

            with pytest.raises(ServerError) as exc_info:
                provider.generate("Test prompt", {})

            assert exc_info.value.code == 503
            # Should be called GEMINI_MAX_RETRIES times
            assert (
                mock_client.models.generate_content.call_count
                == GEMINI_MAX_RETRIES
            )
