"""
Centralized model caching for ML models.

Provides cached loading of SentenceTransformer and spaCy models
to reduce cold start time and memory usage in serverless deployments.
Models are cached per process and reused across requests.
"""

import logging
from functools import _CacheInfo, lru_cache

import spacy
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_spacy_model(model_name: str) -> spacy.Language:
    """
    Load and cache a spaCy model.

    Args:
        model_name: Name of the spaCy model to load (e.g.,
            "en_core_web_sm").

    Returns:
        Loaded spaCy language model.

    Raises:
        RuntimeError: If the specified model cannot be loaded.

    """
    logger.info(f"Loading spaCy model: {model_name}")
    try:
        model = spacy.load(model_name)
    except OSError as err:
        msg = (
            f"spaCy model '{model_name}' not found. "
            f"Install with: python -m spacy download {model_name}"
        )
        raise RuntimeError(msg) from err
    else:
        logger.info(f"spaCy model '{model_name}' loaded and cached")
        return model


@lru_cache(maxsize=4)
def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.

    Uses LRU cache to reuse models across requests within the same
    process.

    Args:
        model_name: Name of the SentenceTransformer model to load.

    Returns:
        Loaded SentenceTransformer model instance.

    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name, backend="onnx")
    logger.info(f"SentenceTransformer model '{model_name}' loaded and cached")
    return model


def clear_model_cache() -> None:
    """
    Clear all cached models.

    Call this to free memory after pipeline processing completes.
    Useful for long-running containers that need to reclaim memory.
    """
    get_spacy_model.cache_clear()
    get_sentence_transformer.cache_clear()
    logger.info("Model cache cleared")


def get_cache_info() -> dict[str, _CacheInfo]:
    """
    Get cache statistics for debugging.

    Returns:
        Dictionary with _CacheInfo for each model type.
        Each _CacheInfo has attributes: hits, misses, maxsize, currsize.

    """
    return {
        "spacy": get_spacy_model.cache_info(),
        "sentence_transformer": get_sentence_transformer.cache_info(),
    }
