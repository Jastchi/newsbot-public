"""
LLM-as-a-Judge Agent.

Validates LLM outputs and rewrites them when violations are detected.
A violation occurs when the LLM adds meta-commentary, uses wrong format,
or goes off-topic.

Uses LLM provider abstraction to support Ollama and Gemini.
"""

import json
import logging
import re

from pydantic import ValidationError

from newsbot.agents.prompts import get_prompt
from newsbot.llm_provider import LLMProvider, get_llm_provider
from newsbot.models import JudgeVerdict, ValidationStats, ViolationType
from utilities import models as config_models

logger = logging.getLogger(__name__)

# Untrusted text must not be able to open or close a prompt delimiter
_DELIMITER_TAG = re.compile(
    r"<\s*(/?)\s*(source_articles|output_to_evaluate|original_text)",
    re.IGNORECASE,
)


class JudgeAgent:
    """
    Agent that validates LLM outputs and rewrites them on violations.

    The judge uses structured output to detect when another LLM has
    produced meta-commentary, wrong format, or off-topic content.
    When violations are found, the output is rewritten to fix the
    issues.
    """

    INVALID_MESSAGE_LENGTH_FOR_LOGGING = 200
    # The verdict JSON is tiny, but Gemini thinking models spend output
    # tokens on thinking first. A low cap leaves no room for the verdict
    # and makes the judge fail open.
    VERDICT_MAX_TOKENS = 2048
    # Source text sent to the judge is cut to this size (head only)
    MAX_SOURCE_CHARS = 12000

    def __init__(
        self,
        config: config_models.ConfigModel,
        stats: ValidationStats | None = None,
    ) -> None:
        """
        Initialize the Judge Agent.

        Args:
            config: Configuration dictionary containing LLM settings.
            stats: Optional shared validation counters.

        """
        self.config = config
        self.stats = stats or ValidationStats()
        llm_config = config.llm

        # Judge-specific configuration
        self.enabled = llm_config.judge_enabled
        self.max_rewrites = llm_config.judge_max_retries
        self.check_source = llm_config.judge_check_source
        self.provider_name = llm_config.provider

        # Use judge_model if specified, otherwise fall back to main
        self.main_model = llm_config.model or "llama2"
        self.judge_model = llm_config.judge_model or self.main_model

        # Initialize LLM provider with judge_model if specified
        # Create a modified config that uses judge_model as the model
        judge_config = self._create_judge_config(config)
        self.provider: LLMProvider = get_llm_provider(judge_config)

        # Load prompt templates
        self.validation_prompt = self._load_prompt("judge_validation.txt")
        self.rewrite_prompt = self._load_prompt("judge_rewrite.txt")
        self.source_block = self._load_prompt("judge_source_block.txt")
        self.claim_rule = self._load_prompt("judge_claim_rule.txt")

        # Load rewrite instruction prompts
        self.rewrite_instructions = {
            ViolationType.META_COMMENTARY: self._load_prompt(
                "judge_rewrite_meta_commentary.txt",
            ),
            ViolationType.WRONG_FORMAT: self._load_prompt(
                "judge_rewrite_wrong_format.txt",
            ),
            ViolationType.OFF_TOPIC: self._load_prompt(
                "judge_rewrite_off_topic.txt",
            ),
            ViolationType.UNSUPPORTED_CLAIM: self._load_prompt(
                "judge_rewrite_unsupported_claim.txt",
            ),
            None: self._load_prompt("judge_rewrite_default.txt"),
        }

        if self.enabled and self.judge_model == self.main_model:
            logger.warning(
                "Judge uses the same model as the writer "
                f"({self.judge_model}); set judge_model to a different "
                "model to avoid self-approval.",
            )

        logger.info(
            f"Judge agent initialized: enabled={self.enabled}, "
            f"model={self.judge_model}, max_rewrites={self.max_rewrites}",
        )

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template for the configured provider."""
        return get_prompt(self.provider_name, filename)

    def _create_judge_config(
        self,
        config: config_models.ConfigModel,
    ) -> config_models.ConfigModel:
        """
        Create a config copy with judge_model as the model.

        If judge_model is specified and different from the main model,
        creates a modified config that uses judge_model. Otherwise
        returns the original config.

        Args:
            config: Original configuration dictionary.

        Returns:
            Config with judge_model substituted for model if specified.

        """
        llm_config = config.llm
        main_model = self.main_model
        judge_model = llm_config.judge_model

        # If no judge_model specified or same as main, use original
        if not judge_model or judge_model == main_model:
            return config

        # Create a new config with judge model
        # Use model_copy with update to properly create a modified copy
        judge_llm_config = llm_config.model_copy(update={"model": judge_model})
        judge_config = config.model_copy(update={"llm": judge_llm_config})

        logger.debug(
            f"Judge using separate model: {judge_model} "
            f"(main model: {main_model})",
        )
        return judge_config

    def validate_output(
        self,
        output: str,
        prompt_context: str,
        *,
        allow_empty: bool = False,
        source: str | None = None,
    ) -> JudgeVerdict:
        """
        Validate an LLM output against the prompt instructions.

        Args:
            output: The LLM output to validate.
            prompt_context: Description of what the original prompt
                asked for.
            allow_empty: If True, empty output is considered valid.
                Use this for prompts where returning nothing is an
                acceptable response.
            source: Source article text to check claims against.
                Ignored when the config disables source checking.

        Returns:
            JudgeVerdict with validation result.

        """
        if not self.enabled:
            return JudgeVerdict(is_valid=True)

        # Check for empty responses
        is_empty = not output or not output.strip()

        if is_empty:
            if allow_empty:
                logger.debug("Judge: empty output allowed by caller")
                return JudgeVerdict(is_valid=True)
            logger.info(
                "Judge: empty output violation. Original: ''",
            )
            return JudgeVerdict(
                is_valid=False,
                violation_type=ViolationType.WRONG_FORMAT,
            )

        self.stats.inc("judge.checked")
        if not self.check_source:
            source = None

        try:
            # Build judge prompt
            judge_prompt = self.validation_prompt.format(
                prompt_context=prompt_context,
                output=self._escape(output),
                source_block=self._source_block(source),
                claim_rule=self.claim_rule if source else "",
            )

            # Call LLM with structured output
            verdict_data = self.provider.chat_json(
                messages=[{"role": "user", "content": judge_prompt}],
                options={
                    "temperature": 0.1,
                    "num_predict": self.VERDICT_MAX_TOKENS,
                },
                schema=JudgeVerdict.model_json_schema(),
            )

            verdict = JudgeVerdict.model_validate(verdict_data)

            # Without a source the judge cannot verify claims
            if (
                not source
                and verdict.violation_type == ViolationType.UNSUPPORTED_CLAIM
            ):
                verdict = JudgeVerdict(is_valid=True)

            if not verdict.is_valid:
                self.stats.inc("judge.violations")
                self.stats.inc(
                    f"judge.violation.{verdict.violation_type or 'unknown'}",
                )
                truncated = output[: self.INVALID_MESSAGE_LENGTH_FOR_LOGGING]
                suffix = (
                    "..."
                    if len(output) > self.INVALID_MESSAGE_LENGTH_FOR_LOGGING
                    else ""
                )
                logger.info(
                    f"Judge: {verdict.violation_type} violation. "
                    "Original: "
                    f"{truncated}{suffix}",
                )

        except (json.JSONDecodeError, ValidationError):
            self.stats.inc("judge.fail_open_parse")
            logger.warning("Failed to parse judge response")
            return JudgeVerdict(is_valid=True)

        except Exception:
            self.stats.inc("judge.fail_open_error")
            logger.exception("Error during judge validation")
            return JudgeVerdict(is_valid=True)

        else:
            return verdict

    def _rewrite_output(
        self,
        output: str,
        prompt_context: str,
        violation_type: ViolationType | None,
        source: str | None = None,
    ) -> str:
        """
        Rewrite an invalid output to fix violations.

        Args:
            output: The invalid LLM output to rewrite.
            prompt_context: Description of what the output should be.
            violation_type: The type of violation detected.
            source: Source article text. Used only to fix unsupported
                claims.

        Returns:
            Rewritten output with violations removed.

        """
        rewrite_instructions = self._get_rewrite_instructions(violation_type)

        # Only claim fixes need the source; the rest save the tokens
        if violation_type != ViolationType.UNSUPPORTED_CLAIM:
            source = None

        prompt = self.rewrite_prompt.format(
            output=self._escape(output),
            prompt_context=prompt_context,
            rewrite_instructions=rewrite_instructions,
            source_block=self._source_block(source),
        )

        try:
            rewritten = self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            logger.debug("Output rewritten successfully")

        except Exception:
            self.stats.inc("judge.rewrite_error")
            logger.exception("Error rewriting output, returning original")
            return output

        return rewritten

    @staticmethod
    def _escape(text: str) -> str:
        """Neutralize prompt delimiters inside untrusted text."""
        return _DELIMITER_TAG.sub(r"&lt;\1\2", text)

    def _source_block(self, source: str | None) -> str:
        """Render the truncated, escaped source block, or "" if none."""
        if not source:
            return ""
        if len(source) > self.MAX_SOURCE_CHARS:
            source = source[: self.MAX_SOURCE_CHARS] + "\n[... truncated ...]"
        return self.source_block.format(source=self._escape(source))

    def _get_rewrite_instructions(
        self,
        violation_type: ViolationType | None,
    ) -> str:
        """Get rewrite instructions based on violation type."""
        return self.rewrite_instructions.get(
            violation_type,
            self.rewrite_instructions[None],
        )

    def validate_and_fix(
        self,
        output: str,
        prompt_context: str,
        *,
        allow_empty: bool = False,
        source: str | None = None,
    ) -> str:
        """
        Validate output and rewrite if violations are found.

        Args:
            output: The LLM output to validate.
            prompt_context: Description of what the output should be.
            allow_empty: If True, empty output is considered valid.
                Use this for prompts where returning nothing is an
                acceptable response.
            source: Source article text to check claims against.

        Returns:
            The original output if valid, or rewritten output if
            invalid.

        """
        if not self.enabled:
            return output

        current_output = output

        for attempt in range(self.max_rewrites + 1):
            verdict = self.validate_output(
                current_output,
                prompt_context,
                allow_empty=allow_empty,
                source=source,
            )

            # Normalize no-content responses to empty string when
            # allowed
            if allow_empty and verdict.is_no_content:
                if not verdict.is_valid:
                    logger.warning(
                        "Judge: no-content verdict flagged invalid",
                    )
                logger.debug(
                    "Judge: no-content response detected, normalizing to "
                    f"empty string. Original: {current_output!r}",
                )
                return ""

            if verdict.is_valid:
                if attempt > 0:
                    logger.info(f"Output fixed after {attempt} rewrite(s)")
                return current_output

            if attempt < self.max_rewrites:
                self.stats.inc("judge.rewrites")
                logger.warning(
                    f"Validation failed ({verdict.violation_type}), "
                    f"rewriting (attempt {attempt + 1}/{self.max_rewrites})",
                )
                current_output = self._rewrite_output(
                    current_output,
                    prompt_context,
                    verdict.violation_type,
                    source,
                )
            else:
                self.stats.inc("judge.max_rewrites_exceeded")
                logger.warning(
                    f"Max rewrites ({self.max_rewrites}) exceeded, "
                    "returning last output",
                )

        return current_output
