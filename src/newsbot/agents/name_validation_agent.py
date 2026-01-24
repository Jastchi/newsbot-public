"""
Named Entity Validation Agent.

Validates that named entities in LLM outputs appear in source articles.
Uses spaCy NER to extract entities and verifies them against source
content. When unverified entities are found, requests LLM to rewrite
the output without them.

NameValidationAgent(config: Config) -> NameValidationAgent
NameValidationAgent.validate_entities(text: str, sources: list[str])
    -> NameValidationResult
NameValidationAgent.validate_and_fix(
    text: str, sources: list[str], prompt_context: str
) -> str

- Entities are extracted using spaCy NER (PERSON, ORG, GPE, LOC, NORP).
- Verification uses multi-tier matching: exact, partial, normalized.
- Unverified entities trigger LLM rewrite up to max_retries times.
- spaCy model configurable via llm.spacy_model (default en_core_web_sm).
"""

import logging
import re

import spacy

from newsbot.agents.prompts import get_prompt
from newsbot.llm_provider import get_llm_provider
from newsbot.model_cache import get_spacy_model
from newsbot.models import NameValidationResult
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

# Entity types to extract and validate
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP"}

# Minimum entity length to consider (avoid single-letter matches)
MIN_ENTITY_LENGTH = 2

# Minimum word length for partial matching (skip articles, prepositions)
MIN_PARTIAL_MATCH_LENGTH = 3

# Default spaCy model for NER
DEFAULT_SPACY_MODEL = "en_core_web_sm"

# Re-export for backward compatibility with existing code and tests
_load_spacy_model = get_spacy_model


class NameValidationAgent:
    """
    Agent that validates named entities in LLM output against sources.

    Extracts named entities using spaCy NER and verifies each appears
    in at least one source article. When unverified entities are found,
    the output is rewritten by the LLM to remove or generalize them.
    """

    def __init__(self, config: ConfigModel) -> None:
        """
        Initialize the Name Validation Agent.

        Args:
            config: Configuration dictionary containing LLM settings.

        """
        self.config = config
        llm_config = config.llm

        # Name validation configuration
        self.enabled = llm_config.name_validation_enabled
        self.max_retries = llm_config.name_validation_max_retries
        self.provider_name = llm_config.provider
        self.spacy_model_name = llm_config.spacy_model or DEFAULT_SPACY_MODEL

        # Initialize LLM provider for rewrites
        self.provider = get_llm_provider(config)

        # Load rewrite prompt template
        self.rewrite_prompt = self._load_prompt("name_validation_rewrite.txt")

        # Lazy-load spaCy model
        self._nlp = None

        logger.info(
            f"Name validation agent initialized: enabled={self.enabled}, "
            f"max_retries={self.max_retries}, "
            f"spacy_model={self.spacy_model_name}",
        )

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template for the configured provider."""
        return get_prompt(self.provider_name, filename)

    @property
    def nlp(self) -> spacy.Language:
        """Lazy-load spaCy model on first access."""
        if self._nlp is None:
            self._nlp = get_spacy_model(self.spacy_model_name)
        return self._nlp

    def extract_entities(self, text: str) -> set[str]:
        """
        Extract named entities from text using spaCy NER.

        Args:
            text: Text to extract entities from.

        Returns:
            Set of entity text strings.

        """
        if not text or not text.strip():
            return set()

        doc = self.nlp(text)
        entities = set()

        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                entity_text = ent.text.strip()
                if len(entity_text) >= MIN_ENTITY_LENGTH:
                    entities.add(entity_text)

        return entities

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Lowercases, removes extra whitespace, and strips punctuation
        from edges.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text string.

        """
        # Lowercase
        normalized = text.lower()
        # Remove leading/trailing punctuation and dashes
        normalized = normalized.strip(".,;:!?\"'()[]{}\u2014\u2013-")
        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _entity_in_source(self, entity: str, source_text: str) -> bool:
        """
        Check if an entity appears in source text.

        Uses multi-tier matching:
        1. Exact match (case-insensitive)
        2. Partial match (any name part appears)
        3. Normalized match (whitespace/punctuation normalized)

        Args:
            entity: Entity text to search for.
            source_text: Source article content.

        Returns:
            True if entity is verified in source.

        """
        if not entity or not source_text:
            return False

        entity_lower = entity.lower()
        source_lower = source_text.lower()

        # Tier 1: Exact match (case-insensitive)
        if entity_lower in source_lower:
            return True

        # Tier 2: Partial match (check each word in entity)
        entity_parts = entity.split()
        if len(entity_parts) > 1:
            # For multi-word entities, check if any significant part
            # appears
            for part in entity_parts:
                # Strip punctuation (dashes: em-dash, en-dash, hyphen)
                part_lower = part.lower().strip(".,;:!?\"'()[]{}\u2014\u2013-")
                # Skip short words (articles, prepositions)
                if (
                    len(part_lower) >= MIN_PARTIAL_MATCH_LENGTH
                    and part_lower in source_lower
                ):
                    return True

        # Tier 3: Normalized match
        normalized_entity = self._normalize_text(entity)
        normalized_source = self._normalize_text(source_text)
        return normalized_entity in normalized_source

    def validate_entities(
        self,
        text: str,
        sources: list[str],
    ) -> NameValidationResult:
        """
        Validate entities in text against source content.

        Args:
            text: Text containing entities to validate.
            sources: List of source article content strings.

        Returns:
            NameValidationResult with verified and unverified entities.

        """
        if not self.enabled:
            return NameValidationResult(is_valid=True)

        if not text or not text.strip():
            return NameValidationResult(is_valid=True)

        # Extract entities from the LLM output
        entities = self.extract_entities(text)

        if not entities:
            return NameValidationResult(is_valid=True)

        # Combine all source content
        combined_sources = " ".join(sources)

        verified = []
        unverified = []

        for entity in entities:
            if self._entity_in_source(entity, combined_sources):
                verified.append(entity)
            else:
                unverified.append(entity)
                logger.debug(
                    f"Unverified entity: '{entity}' not found in sources",
                )

        is_valid = len(unverified) == 0

        if not is_valid:
            logger.info(
                f"Name validation found {len(unverified)} unverified "
                f"entities: {unverified}",
            )

        return NameValidationResult(
            is_valid=is_valid,
            verified_entities=verified,
            unverified_entities=unverified,
        )

    def _rewrite_without_entities(
        self,
        text: str,
        unverified_entities: list[str],
    ) -> str:
        """
        Rewrite text to remove unverified entities.

        Args:
            text: Original text with unverified entities.
            unverified_entities: List of entity names to remove.

        Returns:
            Rewritten text with entities generalized.

        """
        unverified_names = ", ".join(unverified_entities)

        prompt = self.rewrite_prompt.format(
            unverified_names=unverified_names,
            text=text,
        )

        try:
            rewritten = self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 8192},
            )
        except Exception:
            logger.exception(
                "Error rewriting text to remove entities, returning original",
            )
            return text
        else:
            logger.debug(
                f"Rewrote text to remove entities: {unverified_entities}",
            )
            return rewritten

    def validate_and_fix(
        self,
        text: str,
        sources: list[str],
        _prompt_context: str | None = None,
    ) -> str:
        """
        Validate entities and rewrite if unverified entities found.

        Args:
            text: LLM output text to validate.
            sources: List of source article content strings.
            prompt_context: Unused; matches JudgeAgent interface.

        Returns:
            Original text if valid, or rewritten text if not.

        """
        if not self.enabled:
            return text

        if not text or not text.strip():
            return text

        current_text = text

        for attempt in range(self.max_retries + 1):
            result = self.validate_entities(current_text, sources)

            if result.is_valid:
                if attempt > 0:
                    logger.info(
                        f"Names validated after {attempt} rewrite(s)",
                    )
                return current_text

            if attempt < self.max_retries:
                logger.warning(
                    f"Found {len(result.unverified_entities)} unverified "
                    f"entities, rewriting "
                    f"(attempt {attempt + 1}/{self.max_retries})",
                )
                current_text = self._rewrite_without_entities(
                    current_text,
                    result.unverified_entities,
                )
            else:
                logger.warning(
                    f"Max retries ({self.max_retries}) exceeded for name "
                    f"validation. Remaining unverified: "
                    f"{result.unverified_entities}",
                )

        return current_text
