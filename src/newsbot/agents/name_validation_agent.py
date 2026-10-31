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
- Verification matches the entity in the sources, starting at a word
  boundary ("Oman" is not found in "woman"), else by part of its name:
  the surname of a person, or a word of another entity that is not a
  generic one such as "bank" or "the" (listed in
  generic_entity_words.txt).
- Unverified entities trigger LLM rewrite up to max_retries times.
- spaCy model configurable via llm.spacy_model (default en_core_web_sm).
"""

import logging
import re
import unicodedata
from pathlib import Path

import spacy

from newsbot.agents.prompts import get_prompt
from newsbot.llm_provider import get_llm_provider
from newsbot.model_cache import get_spacy_model
from newsbot.models import NameValidationResult, ValidationStats
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

# Entity types to extract and validate
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP"}

# Minimum entity length to consider (avoid single-letter matches)
MIN_ENTITY_LENGTH = 2

# Minimum word length for partial matching (skip articles, prepositions)
MIN_PARTIAL_MATCH_LENGTH = 3

# A single word has no part to match on its own
MIN_PARTS_FOR_PARTIAL_MATCH = 2

# Entities this short must match a whole word: "UN" is not in "until"
MAX_WHOLE_WORD_LENGTH = 3

# Ends of a person's name that are not the surname: Jr, Sr, or a regnal
# number such as the "XIV" of Leo XIV. Capitals only: "Li" is a surname.
# Only the end of a name is checked, so "XI JINPING" keeps both words.
NAME_SUFFIX = re.compile(r"(?i:jr|sr)|[IVXLC]+")

# The possessive spaCy sometimes keeps in an entity: "Rubio's"
POSSESSIVE = re.compile(r"[\u2019']s$", re.IGNORECASE)

# Accents that NFKD splits off; letters it leaves whole are mapped here
COMBINING_MARKS = re.compile(r"[\u0300-\u036f]")
UNACCENTED = str.maketrans(
    {
        "\u0111": "dj",
        "\u0142": "l",
        "\u00f8": "o",
        "\u00e6": "ae",
        "\u0153": "oe",
        "\u00df": "ss",
    },
)

# Words that name a kind of body, office or place, not one of them
GENERIC_WORDS_FILE = Path(__file__).with_name("generic_entity_words.txt")


def _load_generic_words(path: Path = GENERIC_WORDS_FILE) -> frozenset[str]:
    """Read one word per line, skipping blank lines and # comments."""
    lines = (
        line.strip().lower() for line in path.read_text("utf-8").splitlines()
    )
    return frozenset(line for line in lines if line and line[0] != "#")


GENERIC_WORDS = _load_generic_words()

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

    def __init__(
        self,
        config: ConfigModel,
        stats: ValidationStats | None = None,
    ) -> None:
        """
        Initialize the Name Validation Agent.

        Args:
            config: Configuration dictionary containing LLM settings.
            stats: Optional shared validation counters.

        """
        self.config = config
        self.stats = stats or ValidationStats()
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
        return set(self._extract_labelled(text))

    def _extract_labelled(self, text: str) -> dict[str, str]:
        """Extract entities as {text: NER label}."""
        if not text or not text.strip():
            return {}

        doc = self.nlp(text)
        entities: dict[str, str] = {}

        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                entity_text = POSSESSIVE.sub("", ent.text.strip())
                if len(entity_text) >= MIN_ENTITY_LENGTH:
                    entities.setdefault(entity_text, ent.label_)

        return entities

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Lowercases, drops accents, removes extra whitespace, and strips
        punctuation from edges.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text string.

        """
        # Lowercase, and drop accents: some sources write "Vucic"
        decomposed = unicodedata.normalize("NFKD", text.lower())
        normalized = COMBINING_MARKS.sub("", decomposed).translate(UNACCENTED)
        # Remove leading/trailing punctuation, dashes and markdown marks
        normalized = normalized.strip(".,;:!?\"'()[]{}*_\u2014\u2013-")
        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _contains(source: str, needle: str, *, whole: bool = False) -> bool:
        """
        Check if needle occurs in source, starting at a word boundary.

        A short needle, or one asked for as a whole word, must also end
        at a boundary, with an "s" allowed before it. A short acronym
        also matches its dotted form: "US" is found in "U.S.".
        """
        short = len(needle) <= MAX_WHOLE_WORD_LENGTH
        if short and needle.isalpha():
            body = r"\.?".join(needle) + r"\.?"
        else:
            body = re.escape(needle)
        # An "s" may follow: scraped text lacks the "'" of "Winter's"
        end = r"s?(?!\w)" if whole or short else ""
        return re.search(rf"(?<!\w){body}{end}", source) is not None

    @staticmethod
    def _is_generic(word: str) -> bool:
        """Check if a word, or its singular, is on the generic list."""
        return word in GENERIC_WORDS or word.removesuffix("s") in GENERIC_WORDS

    def _entity_in_source(
        self,
        entity: str,
        normalized_source: str,
        label: str,
    ) -> bool:
        """
        Check if an entity appears in source text.

        Matches the whole entity, else for a multi-word entity part of
        its name: a person's surname (the last word, as in Western name
        order; a first name alone says nothing about who is meant), or
        any word of another entity that is not a generic one.

        Args:
            entity: Entity text to search for.
            normalized_source: Source text passed through
                _normalize_text.
            label: spaCy label of the entity.

        Returns:
            True if entity is verified in source.

        """
        normalized_entity = self._normalize_text(entity)
        if not normalized_entity or not normalized_source:
            return False

        if self._contains(normalized_source, normalized_entity):
            return True

        pairs = [(word, self._normalize_text(word)) for word in entity.split()]
        pairs = [(word, part) for word, part in pairs if part]
        if len(pairs) < MIN_PARTS_FOR_PARTIAL_MATCH:
            return False
        parts = [part for _, part in pairs]

        if label == "PERSON":
            # A suffix (Jr, XIV) ends a name; it is not the surname
            while len(pairs) > 1 and NAME_SUFFIX.fullmatch(
                pairs[-1][0].strip(".,"),
            ):
                pairs.pop()
            return self._contains(normalized_source, pairs[-1][1], whole=True)

        # Skip short words (articles, prepositions)
        words = [p for p in parts if len(p) >= MIN_PARTIAL_MATCH_LENGTH]
        distinctive = [w for w in words if not self._is_generic(w)]
        return any(
            self._contains(normalized_source, word)
            for word in distinctive or words
        )

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
        entities = self._extract_labelled(text)
        self.stats.inc("name.checked")

        if not entities:
            return NameValidationResult(is_valid=True)

        # Combine all source content
        combined_sources = self._normalize_text(" ".join(sources))

        verified = []
        unverified = []

        for entity, label in entities.items():
            if self._entity_in_source(entity, combined_sources, label):
                verified.append(entity)
            else:
                unverified.append(entity)
                logger.debug(
                    f"Unverified entity: '{entity}' not found in sources",
                )

        is_valid = len(unverified) == 0

        if not is_valid:
            self.stats.inc("name.violations")
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
                options={"temperature": 0.3},
            )
        except Exception:
            self.stats.inc("name.rewrite_error")
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
                self.stats.inc("name.rewrites")
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
                self.stats.inc("name.max_retries_exceeded")
                logger.warning(
                    f"Max retries ({self.max_retries}) exceeded for name "
                    f"validation. Remaining unverified: "
                    f"{result.unverified_entities}",
                )

        return current_text
