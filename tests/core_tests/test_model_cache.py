"""Tests for the centralized model cache module."""

from unittest.mock import Mock, patch

import pytest

from newsbot.model_cache import (
    clear_model_cache,
    get_cache_info,
    get_sentence_transformer,
    get_spacy_model,
)


class TestSpacyModelCache:
    """Tests for spaCy model caching."""

    def test_model_loads_successfully(self):
        """Test that a valid spaCy model loads."""
        clear_model_cache()

        nlp = get_spacy_model("en_core_web_sm")

        assert nlp is not None
        assert nlp.meta["name"] == "core_web_sm"

    def test_cache_returns_same_instance(self):
        """Test that cached model returns the same instance."""
        clear_model_cache()

        nlp1 = get_spacy_model("en_core_web_sm")
        nlp2 = get_spacy_model("en_core_web_sm")

        assert nlp1 is nlp2

    def test_different_models_cached_separately(self):
        """Test that different model names are cached separately."""
        clear_model_cache()

        nlp_en = get_spacy_model("en_core_web_sm")
        nlp_xx = get_spacy_model("xx_ent_wiki_sm")

        assert nlp_en is not nlp_xx
        assert nlp_en.meta["name"] == "core_web_sm"
        assert nlp_xx.meta["name"] == "ent_wiki_sm"

    def test_invalid_model_raises_runtime_error(self):
        """Test that invalid model name raises RuntimeError."""
        clear_model_cache()

        with pytest.raises(RuntimeError, match="not found"):
            get_spacy_model("nonexistent_model_xyz")


class TestSentenceTransformerCache:
    """Tests for SentenceTransformer model caching."""

    @patch("newsbot.model_cache.SentenceTransformer")
    def test_model_loads_successfully(self, mock_st_class):
        """Test that SentenceTransformer model loads."""
        clear_model_cache()
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        result = get_sentence_transformer("test-model")

        assert result is mock_model
        mock_st_class.assert_called_once_with("test-model", backend="onnx")

    @patch("newsbot.model_cache.SentenceTransformer")
    def test_cache_returns_same_instance(self, mock_st_class):
        """Test that cached model returns the same instance."""
        clear_model_cache()
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        result1 = get_sentence_transformer("test-model")
        result2 = get_sentence_transformer("test-model")

        assert result1 is result2
        # Should only be called once due to caching
        mock_st_class.assert_called_once()

    @patch("newsbot.model_cache.SentenceTransformer")
    def test_different_models_cached_separately(self, mock_st_class):
        """Test that different model names are cached separately."""
        clear_model_cache()
        mock_model_a = Mock(name="model_a")
        mock_model_b = Mock(name="model_b")
        mock_st_class.side_effect = [mock_model_a, mock_model_b]

        result_a = get_sentence_transformer("model-a")
        result_b = get_sentence_transformer("model-b")

        assert result_a is not result_b
        assert mock_st_class.call_count == 2


class TestClearModelCache:
    """Tests for clear_model_cache function."""

    def test_clears_spacy_cache(self):
        """Test that clear_model_cache resets spaCy cache."""
        # Load a model to populate cache
        nlp1 = get_spacy_model("en_core_web_sm")

        # Clear the cache
        clear_model_cache()

        # Check cache info shows empty
        info = get_cache_info()
        assert info["spacy"].currsize == 0

    @patch("newsbot.model_cache.SentenceTransformer")
    def test_clears_sentence_transformer_cache(self, mock_st_class):
        """Test that clear_model_cache resets SentenceTransformer cache."""
        mock_st_class.return_value = Mock()

        # Load a model to populate cache
        get_sentence_transformer("test-model")

        # Clear the cache
        clear_model_cache()

        # Check cache info shows empty
        info = get_cache_info()
        assert info["sentence_transformer"].currsize == 0

    @patch("newsbot.model_cache.SentenceTransformer")
    def test_subsequent_load_creates_new_instance(self, mock_st_class):
        """Test that loading after clear creates a new instance."""
        clear_model_cache()
        mock_model_1 = Mock(name="first")
        mock_model_2 = Mock(name="second")
        mock_st_class.side_effect = [mock_model_1, mock_model_2]

        # First load
        result1 = get_sentence_transformer("test-model")

        # Clear and reload
        clear_model_cache()
        result2 = get_sentence_transformer("test-model")

        # Should be different instances
        assert result1 is not result2
        assert mock_st_class.call_count == 2


class TestGetCacheInfo:
    """Tests for get_cache_info function."""

    def test_returns_both_cache_infos(self):
        """Test that get_cache_info returns info for both caches."""
        clear_model_cache()

        info = get_cache_info()

        assert "spacy" in info
        assert "sentence_transformer" in info

    def test_cache_info_has_required_attributes(self):
        """Test that cache info has the expected attributes."""
        clear_model_cache()

        info = get_cache_info()

        # Check spaCy cache info
        spacy_info = info["spacy"]
        assert hasattr(spacy_info, "hits")
        assert hasattr(spacy_info, "misses")
        assert hasattr(spacy_info, "maxsize")
        assert hasattr(spacy_info, "currsize")

        # Check SentenceTransformer cache info
        st_info = info["sentence_transformer"]
        assert hasattr(st_info, "hits")
        assert hasattr(st_info, "misses")
        assert hasattr(st_info, "maxsize")
        assert hasattr(st_info, "currsize")

    def test_cache_info_tracks_hits_and_misses(self):
        """Test that cache info correctly tracks hits and misses."""
        clear_model_cache()

        # First call = miss
        get_spacy_model("en_core_web_sm")
        info1 = get_cache_info()
        assert info1["spacy"].misses == 1
        assert info1["spacy"].hits == 0

        # Second call = hit
        get_spacy_model("en_core_web_sm")
        info2 = get_cache_info()
        assert info2["spacy"].misses == 1
        assert info2["spacy"].hits == 1

    def test_maxsize_is_configured(self):
        """Test that cache maxsize is set as expected."""
        info = get_cache_info()

        assert info["spacy"].maxsize == 4
        assert info["sentence_transformer"].maxsize == 4
