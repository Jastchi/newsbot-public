"""Tests for Judge Agent.

Tests the LLM-as-a-judge validation and rewrite mechanism.
"""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from newsbot.agents.judge_agent import JudgeAgent
from newsbot.models import JudgeVerdict, ViolationType
from utilities.models import ConfigModel, LLMConfigModel


@pytest.fixture
def judge_config() -> ConfigModel:
    """Configuration for judge agent testing."""
    return ConfigModel(
        llm=LLMConfigModel(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=2000,
            judge_enabled=True,
            judge_model="llama2",
            judge_max_retries=2,
        ),
    )


@pytest.fixture
def judge_config_disabled() -> ConfigModel:
    """Configuration with judge disabled."""
    return ConfigModel(
        llm=LLMConfigModel(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434",
            judge_enabled=False,
        ),
    )


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.chat.return_value = "rewritten output"
    provider.chat_json.return_value = {
        "is_valid": True,
        "violation_type": None,
        "is_no_content": False,
    }
    return provider


class TestJudgeVerdict:
    """Test cases for JudgeVerdict model."""

    def test_valid_verdict(self):
        """Test creating a valid verdict."""
        verdict = JudgeVerdict(is_valid=True)
        assert verdict.is_valid is True
        assert verdict.violation_type is None
        assert verdict.is_no_content is False

    def test_invalid_verdict(self):
        """Test creating an invalid verdict."""
        verdict = JudgeVerdict(
            is_valid=False,
            violation_type=ViolationType.META_COMMENTARY,
        )
        assert verdict.is_valid is False
        assert verdict.violation_type == ViolationType.META_COMMENTARY
        assert verdict.is_no_content is False

    def test_no_content_verdict(self):
        """Test creating a no-content verdict."""
        verdict = JudgeVerdict(
            is_valid=False,
            violation_type=ViolationType.META_COMMENTARY,
            is_no_content=True,
        )
        assert verdict.is_valid is False
        assert verdict.violation_type == ViolationType.META_COMMENTARY
        assert verdict.is_no_content is True

    def test_violation_types(self):
        """Test all violation types."""
        assert ViolationType.META_COMMENTARY.value == "meta_commentary"
        assert ViolationType.WRONG_FORMAT.value == "wrong_format"
        assert ViolationType.OFF_TOPIC.value == "off_topic"


class TestJudgeAgentInit:
    """Test cases for JudgeAgent initialization."""

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_init_with_judge_enabled(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test initialization with judge enabled."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)

        assert agent.enabled is True
        assert agent.max_rewrites == 2
        assert agent.judge_model == "llama2"

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_init_with_judge_disabled(
        self,
        mock_get_provider,
        mock_provider,
        judge_config_disabled,
    ):
        """Test initialization with judge disabled."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config_disabled)

        assert agent.enabled is False

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_init_uses_main_model_as_default_judge(
        self,
        mock_get_provider,
        mock_provider,
    ):
        """Test that judge model defaults to main model."""
        config = ConfigModel(
            llm=LLMConfigModel(
                provider="ollama",
                model="gemma3:4b",
                base_url="http://localhost:11434",
                judge_enabled=True,
            ),
        )
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(config)

        assert agent.judge_model == "gemma3:4b"


class TestJudgeAgentValidation:
    """Test cases for JudgeAgent.validate_output."""

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_returns_valid_for_disabled_judge(
        self,
        mock_get_provider,
        mock_provider,
        judge_config_disabled,
    ):
        """Test that disabled judge always returns valid."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config_disabled)
        verdict = agent.validate_output(
            output="Some output",
            prompt_context="Generate a title",
        )

        assert verdict.is_valid is True

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_empty_output(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test validation of empty output."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="",
            prompt_context="Generate a title",
        )

        assert verdict.is_valid is False
        assert verdict.violation_type == ViolationType.WRONG_FORMAT

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_empty_output_allowed_when_allow_empty_is_true(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test that empty output is valid when allow_empty=True."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="",
            prompt_context="Extract additional points, or nothing if none",
            allow_empty=True,
        )

        assert verdict.is_valid is True
        assert verdict.violation_type is None

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_whitespace_only_output_allowed_when_allow_empty_is_true(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test that whitespace-only output is valid when allow_empty=True."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="   \n\t  ",
            prompt_context="Extract additional points, or nothing if none",
            allow_empty=True,
        )

        assert verdict.is_valid is True
        assert verdict.violation_type is None

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_detects_valid_output(
        self,
        mock_get_provider,
        judge_config,
    ):
        """Test validation passes for valid output."""
        mock_provider = MagicMock()
        mock_provider.chat_json.return_value = {
            "is_valid": True,
            "violation_type": None,
        }
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="AI Makes Major Breakthrough in Research",
            prompt_context="Generate a headline",
        )

        assert verdict.is_valid is True

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_detects_meta_commentary(
        self,
        mock_get_provider,
        judge_config,
    ):
        """Test validation detects meta-commentary."""
        mock_provider = MagicMock()
        mock_provider.chat_json.return_value = {
            "is_valid": False,
            "violation_type": "meta_commentary",
        }
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="Here is the headline: AI Makes Breakthrough",
            prompt_context="Generate a headline",
        )

        assert verdict.is_valid is False
        assert verdict.violation_type == ViolationType.META_COMMENTARY

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_handles_json_error(
        self,
        mock_get_provider,
        judge_config,
    ):
        """Test validation handles JSON parsing errors gracefully."""
        mock_provider = MagicMock()
        mock_provider.chat_json.side_effect = Exception("JSON parse error")
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="Some output",
            prompt_context="Generate something",
        )

        assert verdict.is_valid is True

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_handles_exception(
        self,
        mock_get_provider,
        judge_config,
    ):
        """Test validation handles exceptions gracefully."""
        mock_provider = MagicMock()
        mock_provider.chat_json.side_effect = Exception("Connection error")
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="Some output",
            prompt_context="Generate something",
        )

        assert verdict.is_valid is True


class TestJudgeAgentValidateAndFix:
    """Test cases for JudgeAgent.validate_and_fix."""

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_and_fix_returns_original_when_disabled(
        self,
        mock_get_provider,
        mock_provider,
        judge_config_disabled,
    ):
        """Test that disabled judge returns original output."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config_disabled)
        result = agent.validate_and_fix(
            output="Some output",
            prompt_context="Generate something",
        )

        assert result == "Some output"

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_and_fix_returns_original_when_valid(
        self,
        mock_get_provider,
        judge_config,
    ):
        """Test returns original output when validation passes."""
        mock_provider = MagicMock()
        mock_provider.chat_json.return_value = {
            "is_valid": True,
            "violation_type": None,
        }
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        result = agent.validate_and_fix(
            output="Valid headline",
            prompt_context="Generate a headline",
        )

        assert result == "Valid headline"

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_and_fix_rewrites_on_invalid(
        self,
        mock_get_provider,
        judge_config,
    ):
        """Test output is rewritten when validation fails."""
        mock_provider = MagicMock()
        # First call: validation fails; Second call: rewrite; Third: valid
        mock_provider.chat_json.side_effect = [
            {
                "is_valid": False,
                "violation_type": "meta_commentary",
            },
            {
                "is_valid": True,
                "violation_type": None,
            },
        ]
        mock_provider.chat.return_value = "Fixed headline without preamble"
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        result = agent.validate_and_fix(
            output="Here is the headline: Invalid",
            prompt_context="Generate a headline",
        )

        assert result == "Fixed headline without preamble"

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_and_fix_returns_last_after_max_rewrites(
        self,
        mock_get_provider,
        judge_config,
    ):
        """Test returns last output when max rewrites exceeded."""
        mock_provider = MagicMock()
        # Always return invalid verdict
        mock_provider.chat_json.return_value = {
            "is_valid": False,
            "violation_type": "meta_commentary",
        }
        mock_provider.chat.return_value = "Still invalid output"
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        result = agent.validate_and_fix(
            output="Original output",
            prompt_context="Generate a headline",
        )

        # Should return something after max rewrites
        assert isinstance(result, str)

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_and_fix_returns_empty_when_allow_empty_is_true(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test that empty output is returned when allow_empty=True."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        result = agent.validate_and_fix(
            output="",
            prompt_context="Extract additional points, or nothing if none",
            allow_empty=True,
        )

        assert result == ""
        # Should not call chat (no rewrite needed)
        mock_provider.chat.assert_not_called()


class TestJudgeAgentRewriteInstructions:
    """Test cases for rewrite instructions generation."""

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_rewrite_instructions_for_meta_commentary(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test rewrite instructions for meta-commentary."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        instructions = agent._get_rewrite_instructions(
            ViolationType.META_COMMENTARY,
        )

        assert "meta-commentary" in instructions.lower()
        assert "This is" in instructions or "Here is" in instructions

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_rewrite_instructions_for_wrong_format(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test rewrite instructions for wrong format."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        instructions = agent._get_rewrite_instructions(
            ViolationType.WRONG_FORMAT
        )

        assert "format" in instructions.lower()

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_rewrite_instructions_for_off_topic(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test rewrite instructions for off-topic."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        instructions = agent._get_rewrite_instructions(ViolationType.OFF_TOPIC)

        assert (
            "unrelated" in instructions.lower()
            or "relevant" in instructions.lower()
        )

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_rewrite_instructions_for_none(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test rewrite instructions for None violation type."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        instructions = agent._get_rewrite_instructions(None)

        assert (
            "issues" in instructions.lower()
            or "requirements" in instructions.lower()
        )


class TestNoContentResponses:
    """Test cases for handling no-content meta-responses."""

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_parenthetical_no_content_treated_as_valid_empty(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test that '(no new information)' is valid when allow_empty=True."""
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(judge_config)
        verdict = agent.validate_output(
            output="(no new information to add)",
            prompt_context="Extract additional points, or nothing if none",
            allow_empty=True,
        )

        assert verdict.is_valid is True

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_validate_and_fix_normalizes_no_content_to_empty(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test that no-content responses are normalized to empty string."""
        mock_get_provider.return_value = mock_provider
        # Mock judge detecting no-content response
        mock_provider.chat_json.return_value = {
            "is_valid": False,
            "violation_type": "meta_commentary",
            "is_no_content": True,
        }

        agent = JudgeAgent(judge_config)
        result = agent.validate_and_fix(
            output="no additional points",
            prompt_context="Extract additional points, or nothing if none",
            allow_empty=True,
        )

        assert result == ""

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_judge_detects_no_content_responses(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test that judge detects various no-content responses via structured output."""
        mock_get_provider.return_value = mock_provider

        no_content_outputs = [
            "no additional points",
            "nothing interesting to add",
            "nothing to add",
            "no interesting points",
            "no new information",
        ]

        agent = JudgeAgent(judge_config)

        for output in no_content_outputs:
            # Mock judge detecting no-content
            mock_provider.chat_json.return_value = {
                "is_valid": False,
                "violation_type": "meta_commentary",
                "is_no_content": True,
            }

            verdict = agent.validate_output(
                output=output,
                prompt_context="Extract additional points, or nothing if none",
                allow_empty=True,
            )

            assert verdict.is_no_content is True, f"Failed for: {output}"
            assert verdict.is_valid is False
            assert verdict.violation_type == ViolationType.META_COMMENTARY

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_judge_does_not_flag_real_content_as_no_content(
        self,
        mock_get_provider,
        mock_provider,
        judge_config,
    ):
        """Test that judge does not flag actual content as no-content."""
        mock_get_provider.return_value = mock_provider

        real_content_outputs = [
            "The president announced new policies today.",
            "- Bullet point with actual information",
            "Breaking news: Market reaches new high",
        ]

        agent = JudgeAgent(judge_config)

        for output in real_content_outputs:
            # Mock judge detecting valid content
            mock_provider.chat_json.return_value = {
                "is_valid": True,
                "violation_type": None,
                "is_no_content": False,
            }

            verdict = agent.validate_output(
                output=output,
                prompt_context="Extract additional points, or nothing if none",
                allow_empty=True,
            )

            assert verdict.is_no_content is False, f"Wrongly flagged: {output}"
            assert verdict.is_valid is True


class TestJudgeModelConfig:
    """Test cases for judge_model configuration."""

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_judge_uses_separate_model_when_specified(
        self,
        mock_get_provider,
        mock_provider,
    ):
        """Test that judge creates config with judge_model when different."""
        from utilities.models import ConfigModel, LLMConfigModel

        config = ConfigModel(
            llm=LLMConfigModel(
                provider="ollama",
                model="llama3:8b",
                base_url="http://localhost:11434",
                judge_enabled=True,
                judge_model="llama3:1b",
            ),
        )
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(config)

        # Verify the provider was created with judge_model
        call_args = mock_get_provider.call_args[0][0]
        assert call_args.llm.model == "llama3:1b"
        assert agent.judge_model == "llama3:1b"

    @patch("newsbot.agents.judge_agent.get_llm_provider")
    def test_judge_uses_main_model_when_same(
        self,
        mock_get_provider,
        mock_provider,
    ):
        """Test that judge uses main config when judge_model == model."""
        config = ConfigModel(
            llm=LLMConfigModel(
                provider="ollama",
                model="llama3:8b",
                base_url="http://localhost:11434",
                judge_enabled=True,
                judge_model="llama3:8b",  # Same as main model
            ),
        )
        mock_get_provider.return_value = mock_provider

        agent = JudgeAgent(config)

        # Verify the provider was created with original config
        call_args = mock_get_provider.call_args[0][0]
        assert call_args.llm.model == "llama3:8b"
