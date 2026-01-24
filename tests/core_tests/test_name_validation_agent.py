"""
Tests for NameValidationAgent.

Tests entity extraction, verification against source content,
and the rewrite flow for unverified names.
"""

from unittest.mock import MagicMock, patch

import pytest

from newsbot.agents.name_validation_agent import (
    DEFAULT_SPACY_MODEL,
    MIN_ENTITY_LENGTH,
    NameValidationAgent,
)
from newsbot.models import NameValidationResult


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    from utilities.models import ConfigModel, LLMConfigModel
    return ConfigModel(
        llm=LLMConfigModel(
            provider="gemini",
            model="gemini-2.0-flash",
            temperature=0.7,
            max_tokens=2000,
            name_validation_enabled=True,
            name_validation_max_retries=2,
        ),
    )


@pytest.fixture
def mock_spacy_doc():
    """Create a mock spaCy doc with entities."""

    class MockEntity:
        def __init__(self, text: str, label: str):
            self.text = text
            self.label_ = label

    class MockDoc:
        def __init__(self, entities: list[tuple[str, str]]):
            self.ents = [MockEntity(text, label) for text, label in entities]

    return MockDoc


@pytest.fixture
def agent_with_mocked_spacy(mock_config, mock_spacy_doc):
    """Create agent with mocked spaCy model."""
    with (
        patch(
            "newsbot.agents.name_validation_agent._load_spacy_model"
        ) as mock_load,
        patch(
            "newsbot.agents.name_validation_agent.get_llm_provider"
        ) as mock_provider,
    ):
        # Create a mock NLP model
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp

        # Create mock LLM provider
        mock_llm = MagicMock()
        mock_provider.return_value = mock_llm

        agent = NameValidationAgent(mock_config)
        agent._nlp = mock_nlp
        agent._mock_spacy_doc = mock_spacy_doc

        yield agent, mock_nlp, mock_llm


class TestNameValidationAgent:
    """Tests for NameValidationAgent initialization and configuration."""

    def test_agent_initialization(self, mock_config):
        """Test agent initializes with correct configuration."""
        with (
            patch(
                "newsbot.agents.name_validation_agent._load_spacy_model"
            ),
            patch(
                "newsbot.agents.name_validation_agent.get_llm_provider"
            ),
        ):
            agent = NameValidationAgent(mock_config)

            assert agent.enabled is True
            assert agent.max_retries == 2
            assert agent.provider_name == "gemini"

    def test_agent_disabled_when_configured(self, mock_config):
        """Test agent respects disabled configuration."""
        from utilities.models import LLMConfigModel
        mock_config = mock_config.model_copy(
            update={
                "llm": mock_config.llm.model_copy(
                    update={"name_validation_enabled": False}
                )
            }
        )

        with (
            patch(
                "newsbot.agents.name_validation_agent._load_spacy_model"
            ),
            patch(
                "newsbot.agents.name_validation_agent.get_llm_provider"
            ),
        ):
            agent = NameValidationAgent(mock_config)

            assert agent.enabled is False

    def test_agent_uses_default_spacy_model(self, mock_config):
        """Test agent uses default spaCy model when not configured."""
        from utilities.models import LLMConfigModel
        # Create config without spacy_model set (use default)
        config = mock_config.model_copy(
            update={
                "llm": mock_config.llm.model_copy(update={"spacy_model": ""})
            }
        )
        with (
            patch(
                "newsbot.agents.name_validation_agent._load_spacy_model"
            ),
            patch(
                "newsbot.agents.name_validation_agent.get_llm_provider"
            ),
        ):
            agent = NameValidationAgent(config)

            assert agent.spacy_model_name == DEFAULT_SPACY_MODEL

    def test_agent_uses_configured_spacy_model(self, mock_config):
        """Test agent uses spaCy model from config."""
        from utilities.models import LLMConfigModel
        mock_config = mock_config.model_copy(
            update={
                "llm": mock_config.llm.model_copy(
                    update={"spacy_model": "xx_ent_wiki_sm"}
                )
            }
        )

        with (
            patch(
                "newsbot.agents.name_validation_agent._load_spacy_model"
            ),
            patch(
                "newsbot.agents.name_validation_agent.get_llm_provider"
            ),
        ):
            agent = NameValidationAgent(mock_config)

            assert agent.spacy_model_name == "xx_ent_wiki_sm"


class TestEntityExtraction:
    """Tests for named entity extraction."""

    def test_extract_person_entities(self, agent_with_mocked_spacy, mock_spacy_doc):
        """Test extraction of PERSON entities."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("John Smith", "PERSON"),
            ("Jane Doe", "PERSON"),
        ])

        entities = agent.extract_entities("John Smith met with Jane Doe.")

        assert "John Smith" in entities
        assert "Jane Doe" in entities

    def test_extract_organization_entities(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test extraction of ORG entities."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("Microsoft", "ORG"),
            ("United Nations", "ORG"),
        ])

        entities = agent.extract_entities("Microsoft and United Nations.")

        assert "Microsoft" in entities
        assert "United Nations" in entities

    def test_extract_location_entities(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test extraction of GPE and LOC entities."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("New York", "GPE"),
            ("Mount Everest", "LOC"),
        ])

        entities = agent.extract_entities("New York near Mount Everest.")

        assert "New York" in entities
        assert "Mount Everest" in entities

    def test_extract_norp_entities(self, agent_with_mocked_spacy, mock_spacy_doc):
        """Test extraction of NORP (nationalities, religious groups)."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("American", "NORP"),
            ("Democrats", "NORP"),
        ])

        entities = agent.extract_entities("American Democrats voted.")

        assert "American" in entities
        assert "Democrats" in entities

    def test_ignore_short_entities(self, agent_with_mocked_spacy, mock_spacy_doc):
        """Test that entities shorter than MIN_ENTITY_LENGTH are ignored."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("A", "PERSON"),  # Too short
            ("Jo", "PERSON"),  # At minimum length
        ])

        entities = agent.extract_entities("A and Jo were there.")

        assert "A" not in entities
        # Jo should be included if MIN_ENTITY_LENGTH is 2
        if MIN_ENTITY_LENGTH <= 2:
            assert "Jo" in entities

    def test_ignore_non_target_entity_types(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test that non-target entity types are ignored."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("$100", "MONEY"),
            ("January 2024", "DATE"),
            ("John Smith", "PERSON"),
        ])

        entities = agent.extract_entities("$100 in January 2024 by John Smith.")

        assert "John Smith" in entities
        assert "$100" not in entities
        assert "January 2024" not in entities

    def test_empty_text_returns_empty_set(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test empty text returns empty entity set."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([])

        assert agent.extract_entities("") == set()
        assert agent.extract_entities("   ") == set()


class TestEntityVerification:
    """Tests for entity verification against source content."""

    def test_exact_match_verification(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test exact match verification succeeds."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("Benjamin Netanyahu", "PERSON"),
        ])

        sources = ["Benjamin Netanyahu announced the policy."]
        result = agent.validate_entities(
            "Benjamin Netanyahu made a statement.",
            sources,
        )

        assert result.is_valid is True
        assert "Benjamin Netanyahu" in result.verified_entities

    def test_case_insensitive_verification(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test case-insensitive matching works."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("JOHN SMITH", "PERSON"),
        ])

        sources = ["john smith was present."]
        result = agent.validate_entities("JOHN SMITH spoke.", sources)

        assert result.is_valid is True
        assert "JOHN SMITH" in result.verified_entities

    def test_partial_match_verification(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test partial match (last name only) works."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("Benjamin Netanyahu", "PERSON"),
        ])

        sources = ["Prime Minister Netanyahu spoke today."]
        result = agent.validate_entities(
            "Benjamin Netanyahu made a statement.",
            sources,
        )

        assert result.is_valid is True
        assert "Benjamin Netanyahu" in result.verified_entities

    def test_unverified_entity_detection(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test unverified entities are detected."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("John Smith", "PERSON"),
            ("Fake Person", "PERSON"),
        ])

        sources = ["John Smith was mentioned in the article."]
        result = agent.validate_entities(
            "John Smith and Fake Person were there.",
            sources,
        )

        assert result.is_valid is False
        assert "John Smith" in result.verified_entities
        assert "Fake Person" in result.unverified_entities

    def test_multiple_sources_checked(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test all sources are checked for entity matches."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("Person A", "PERSON"),
            ("Person B", "PERSON"),
        ])

        sources = [
            "Person A mentioned here.",
            "Person B mentioned in second source.",
        ]
        result = agent.validate_entities(
            "Person A and Person B were present.",
            sources,
        )

        assert result.is_valid is True
        assert "Person A" in result.verified_entities
        assert "Person B" in result.verified_entities


class TestValidateAndFix:
    """Tests for the validate_and_fix workflow."""

    def test_valid_output_returned_unchanged(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test valid output is returned without modification."""
        agent, mock_nlp, _ = agent_with_mocked_spacy

        mock_nlp.return_value = mock_spacy_doc([
            ("John Smith", "PERSON"),
        ])

        text = "John Smith announced the policy."
        sources = ["John Smith made a statement yesterday."]

        result = agent.validate_and_fix(text, sources)

        assert result == text

    def test_rewrite_triggered_for_unverified_entities(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test LLM rewrite is triggered for unverified entities."""
        agent, mock_nlp, mock_llm = agent_with_mocked_spacy

        # First call returns unverified entity
        # Second call (after rewrite) returns no entities
        call_count = [0]

        def mock_nlp_call(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_spacy_doc([("Fake Name", "PERSON")])
            return mock_spacy_doc([])

        mock_nlp.side_effect = mock_nlp_call
        mock_llm.chat.return_value = "A suspect announced the policy."

        text = "Fake Name announced the policy."
        sources = ["The policy was announced today."]

        result = agent.validate_and_fix(text, sources)

        assert result == "A suspect announced the policy."
        mock_llm.chat.assert_called_once()

    def test_max_retries_respected(self, agent_with_mocked_spacy, mock_spacy_doc):
        """Test max retries limit is respected."""
        agent, mock_nlp, mock_llm = agent_with_mocked_spacy
        agent.max_retries = 2

        # Always return unverified entity
        mock_nlp.return_value = mock_spacy_doc([("Fake Name", "PERSON")])
        mock_llm.chat.return_value = "Still has Fake Name in text."

        text = "Fake Name announced the policy."
        sources = ["The policy was announced today."]

        result = agent.validate_and_fix(text, sources)

        # Should have been called max_retries times
        assert mock_llm.chat.call_count == 2

    def test_disabled_agent_returns_unchanged(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test disabled agent returns text unchanged."""
        agent, mock_nlp, _ = agent_with_mocked_spacy
        agent.enabled = False

        mock_nlp.return_value = mock_spacy_doc([("Fake Name", "PERSON")])

        text = "Fake Name announced the policy."
        sources = ["The policy was announced today."]

        result = agent.validate_and_fix(text, sources)

        assert result == text

    def test_empty_text_returns_unchanged(
        self, agent_with_mocked_spacy, mock_spacy_doc
    ):
        """Test empty text is returned unchanged."""
        agent, _, _ = agent_with_mocked_spacy

        assert agent.validate_and_fix("", ["source"]) == ""
        assert agent.validate_and_fix("  ", ["source"]) == "  "


class TestNormalization:
    """Tests for text normalization."""

    def test_normalize_removes_punctuation(self, agent_with_mocked_spacy):
        """Test normalization removes edge punctuation."""
        agent, _, _ = agent_with_mocked_spacy

        assert agent._normalize_text("Hello!") == "hello"
        assert agent._normalize_text('"Quote"') == "quote"
        assert agent._normalize_text("(Parentheses)") == "parentheses"

    def test_normalize_handles_whitespace(self, agent_with_mocked_spacy):
        """Test normalization handles extra whitespace."""
        agent, _, _ = agent_with_mocked_spacy

        assert agent._normalize_text("Multiple   Spaces") == "multiple spaces"
        assert agent._normalize_text("  Trimmed  ") == "trimmed"


class TestNameValidationResult:
    """Tests for NameValidationResult model."""

    def test_valid_result(self):
        """Test valid result creation."""
        result = NameValidationResult(
            is_valid=True,
            verified_entities=["John Smith", "Microsoft"],
            unverified_entities=[],
        )

        assert result.is_valid is True
        assert len(result.verified_entities) == 2
        assert len(result.unverified_entities) == 0

    def test_invalid_result(self):
        """Test invalid result creation."""
        result = NameValidationResult(
            is_valid=False,
            verified_entities=["John Smith"],
            unverified_entities=["Fake Person"],
        )

        assert result.is_valid is False
        assert "John Smith" in result.verified_entities
        assert "Fake Person" in result.unverified_entities

    def test_default_values(self):
        """Test default empty lists."""
        result = NameValidationResult(is_valid=True)

        assert result.verified_entities == []
        assert result.unverified_entities == []


class TestSpacyModelIntegration:
    """Integration tests that use the real spaCy model.

    These tests verify the spaCy model is installed and works correctly.
    They do NOT mock the spaCy model loading.
    """

    def test_spacy_model_loads(self):
        """Test that the default spaCy model can be loaded."""
        from newsbot.agents.name_validation_agent import _load_spacy_model

        # Clear the cache to force a fresh load
        _load_spacy_model.cache_clear()

        nlp = _load_spacy_model(DEFAULT_SPACY_MODEL)

        assert nlp is not None
        assert nlp.meta["name"] == "core_web_sm"

    def test_multilingual_model_loads(self):
        """Test that the multilingual spaCy model can be loaded."""
        from newsbot.agents.name_validation_agent import _load_spacy_model

        _load_spacy_model.cache_clear()

        nlp = _load_spacy_model("xx_ent_wiki_sm")

        assert nlp is not None
        assert nlp.meta["name"] == "ent_wiki_sm"

    def test_invalid_model_raises_error(self):
        """Test that loading an invalid model raises RuntimeError."""
        from newsbot.agents.name_validation_agent import _load_spacy_model

        _load_spacy_model.cache_clear()

        with pytest.raises(RuntimeError, match="not found"):
            _load_spacy_model("nonexistent_model_xyz")

    def test_real_entity_extraction(self):
        """Test entity extraction with real spaCy model."""
        from newsbot.agents.name_validation_agent import _load_spacy_model

        _load_spacy_model.cache_clear()
        nlp = _load_spacy_model(DEFAULT_SPACY_MODEL)

        text = "President Joe Biden met with Prime Minister Netanyahu in Washington."
        doc = nlp(text)

        # Extract entity texts and labels
        entities = {(ent.text, ent.label_) for ent in doc.ents}

        # Should find at least some of these entities
        entity_texts = {ent.text for ent in doc.ents}
        assert any(
            name in entity_texts
            for name in ["Joe Biden", "Biden", "Netanyahu", "Washington"]
        ), f"Expected known entities, got: {entity_texts}"

    def test_real_validation_flow(self, mock_config):
        """Test full validation flow with real spaCy model."""
        with patch(
            "newsbot.agents.name_validation_agent.get_llm_provider"
        ) as mock_provider:
            mock_llm = MagicMock()
            mock_provider.return_value = mock_llm

            agent = NameValidationAgent(mock_config)

            # Text with a name that IS in the source
            text = "President Biden announced the policy."
            sources = ["President Joe Biden made a statement yesterday."]

            result = agent.validate_entities(text, sources)

            # Biden should be verified since it's in the source
            assert result.is_valid is True or "Biden" in result.verified_entities

    def test_real_unverified_entity_flagged(self, mock_config):
        """Test that an invented name is flagged as unverified."""
        with patch(
            "newsbot.agents.name_validation_agent.get_llm_provider"
        ) as mock_provider:
            mock_llm = MagicMock()
            mock_provider.return_value = mock_llm

            agent = NameValidationAgent(mock_config)

            # Text with a name that is NOT in the source
            text = "Dr. Xylophone McFakename announced the policy."
            sources = ["The government announced a new policy today."]

            result = agent.validate_entities(text, sources)

            # The invented name should be unverified
            assert result.is_valid is False, (
                f"Expected invalid result, got valid. "
                f"Verified: {result.verified_entities}"
            )
            assert len(result.unverified_entities) > 0, (
                "Expected at least one unverified entity"
            )

