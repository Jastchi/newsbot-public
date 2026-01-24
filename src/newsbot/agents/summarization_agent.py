"""
Summarization Agent.

Uses LLM provider abstraction to summarize news articles.
Supports Ollama (local) and Gemini (cloud) providers.
Validates named entities in LLM output against source content.
"""

import logging

from newsbot.agents.judge_agent import JudgeAgent
from newsbot.agents.name_validation_agent import NameValidationAgent
from newsbot.agents.prompts import get_prompt
from newsbot.agents.story_clustering_agent import Story
from newsbot.constants import (
    MAX_ARTICLE_CONTENT_LENGTH,
    MAX_ARTICLE_LENGTH_FALLBACK,
)
from newsbot.llm_provider import LLMProvider, get_llm_provider
from newsbot.models import Article
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)


class SummarizationAgent:
    """
    Agent responsible for summarizing news articles.

    Uses the configured LLM provider (Ollama or Gemini).
    """

    def __init__(self, config: ConfigModel) -> None:
        """
        Initialize the Summarization Agent.

        Args:
            config: Configuration dictionary

        """
        self.config = config
        self.llm_config = config.llm
        self.temperature = self.llm_config.temperature
        self.max_tokens = self.llm_config.max_tokens
        self.provider_name = self.llm_config.provider

        # Initialize LLM provider
        self.provider: LLMProvider = get_llm_provider(config)
        logger.info("Summarization agent initialized with LLM provider")

        # Two-pass summarization configuration
        summarization_config = config.summarization
        self.two_pass_enabled = summarization_config.two_pass_enabled
        self.max_articles_batch = int(summarization_config.max_articles_batch)
        self.article_order = summarization_config.article_order
        logger.info(
            f"Two-pass summarization enabled: {self.two_pass_enabled}, "
            f"max_articles_batch: {self.max_articles_batch}, "
            f"article_order: {self.article_order}",
        )

        # Initialize judge agent for output validation
        self.judge_agent = JudgeAgent(config)

        # Initialize name validation agent for entity verification
        self.name_validation_agent = NameValidationAgent(config)

    def summarize_article(self, article: Article) -> str:
        """
        Summarize a single article.

        Args:
            article: Article object to summarize

        Returns:
            Summary text

        """
        if not article.content:
            logger.warning(
                f"Article '{article.title}' at {article.url} has no content "
                "to summarise.",
            )
            return "Content not available."
        try:
            prompt = self._create_summarization_prompt(article)

            summary = self.provider.generate(
                prompt,
                {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )

            # Validate and fix if needed
            prompt_context = (
                "A 2-3 sentence summary of the article. ONLY the summary "
                "text, no preambles or labels."
            )
            summary = self.judge_agent.validate_and_fix(
                summary,
                prompt_context,
            )

            # Validate named entities against source article
            summary = self.name_validation_agent.validate_and_fix(
                summary,
                [article.content, article.title],
            )

            logger.debug(f"Summarized article: {article.title[:50]}...")

        except Exception:
            logger.exception(f"Error summarizing article '{article.title}'")
            return (
                article.content[:MAX_ARTICLE_LENGTH_FALLBACK] + "..."
                if len(article.content) > MAX_ARTICLE_LENGTH_FALLBACK
                else article.content
            )
        else:
            return summary

    def summarize_story(
        self,
        story: "Story",
    ) -> dict[str, str]:
        """
        Summarize a story using two-pass approach if enabled.

        Args:
            story: Story object containing articles to summarize

        Returns:
            dictionary mapping article URLs to summaries

        """
        logger.info(
            f"Summarizing story '{story.title}' with "
            f"{len(story.articles)} articles",
        )

        if self.two_pass_enabled:
            return self._summarize_story_multi_pass(story)
        return self.summarize_articles_batch(story.articles)

    def _summarize_story_multi_pass(
        self,
        story: Story,
    ) -> dict[str, str]:
        """
        Build story summary and extract additional article summaries.

        Uses a multi-pass approach: build summary, refine, extract
        per-article points, then aggregate by source.

        Args:
            story: Story object containing articles

        Returns:
            dictionary mapping article URLs to summaries

        """
        # Pass 1: Build story summary from articles
        logger.info(
            f"Pass 1: Building story summary from "
            f"{len(story.articles)} articles",
        )
        story_summary = self._build_story_summary(story)
        logger.info("Story summary created")

        # Pass 2: Refine story summary
        logger.info("Pass 2: Refining story summary")
        story_summary = self._refine_story_summary(story_summary, story)
        story.story_summary = story_summary
        logger.info("Story summary refined")

        # Pass 3: Extract additional points from each article
        summaries = self._extract_additional_points_per_article(
            story,
            story_summary,
        )

        # Pass 4: Aggregate additional points by source and summarize
        self._aggregate_additional_points_by_source(story, story_summary)

        logger.info("Completed multi-pass summarization of story")
        return summaries

    def _extract_additional_points_per_article(
        self,
        story: Story,
        story_summary: str,
    ) -> dict[str, str]:
        """
        Extract additional points from each article in the story.

        Args:
            story: Story object containing articles
            story_summary: The cumulative story summary

        Returns:
            dictionary mapping article URLs to additional points

        """
        logger.info("Pass 3: Extracting additional points from each article")
        summaries = {}

        # Sort articles oldest first to match Pass 1 ordering
        sorted_articles = self._sort_articles(story.articles)

        for i, article in enumerate(sorted_articles):
            try:
                additional_points = self._extract_additional_points(
                    article,
                    story_summary,
                )
                summaries[article.url] = additional_points
                article.summary = additional_points

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Processed {i + 1}/{len(story.articles)} articles",
                    )

            except Exception:
                logger.exception(
                    f"Error extracting points from article '{article.title}'",
                )
                summaries[article.url] = article.content[
                    :MAX_ARTICLE_LENGTH_FALLBACK
                ]

        return summaries

    def _aggregate_additional_points_by_source(
        self,
        story: Story,
        story_summary: str,
    ) -> None:
        """
        Aggregate additional points by source and create summaries.

        Args:
            story: Story object containing articles with extracted
                additional points in article.summary
            story_summary: The cumulative story summary

        """
        logger.info("Pass 4: Aggregating additional points by source")
        source_additional_points = {}

        # Group articles by source
        articles_by_source: dict[str, list[Article]] = {}
        for article in story.articles:
            if article.source not in articles_by_source:
                articles_by_source[article.source] = []
            articles_by_source[article.source].append(article)

        # Summarize additional points for each source
        for source, articles_for_source in articles_by_source.items():
            try:
                # Collect additional points from all articles of this
                # source
                source_points = [
                    article.summary
                    for article in articles_for_source
                    if article.summary
                ]

                if source_points:
                    source_summary = self._summarize_source_additional_points(
                        source,
                        source_points,
                        story_summary,
                        articles_for_source,
                    )
                    source_additional_points[source] = source_summary

                    if (len(source_additional_points)) % 5 == 0:
                        logger.info(
                            f"Processed {len(source_additional_points)}/"
                            f"{len(articles_by_source)} sources",
                        )

            except Exception:
                logger.exception(
                    "Error summarizing additional points for source "
                    f"'{source}'",
                )
                # Fallback: use first article's summary if available
                if articles_for_source and articles_for_source[0].summary:
                    source_additional_points[source] = articles_for_source[
                        0
                    ].summary
                else:
                    source_additional_points[source] = ""

        # Store source-level summaries in story object
        story.source_additional_points = source_additional_points

    def _build_story_summary(self, story: "Story") -> str:
        """
        Build a cumulative story summary.

        The summary is built by processing articles sequentially.
        Only the final result is validated.

        Args:
            story: Story object containing articles to summarize

        Returns:
            Cumulative story summary

        """
        articles = self._sort_articles(story.articles)

        story_summary = ""
        total_articles = len(articles)

        for batch_start in range(0, total_articles, self.max_articles_batch):
            batch_end = min(
                batch_start + self.max_articles_batch,
                total_articles,
            )
            batch = articles[batch_start:batch_end]

            for article in batch:
                try:
                    if not article.content:
                        logger.warning(
                            f"Article '{article.title}' has no content",
                        )
                        continue

                    prompt = self._create_story_summary_prompt(
                        article,
                        story_summary,
                    )

                    story_summary = self.provider.generate(
                        prompt,
                        {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                    )
                    logger.debug(
                        f"Updated story summary with: {article.title[:50]}...",
                    )

                except Exception:
                    logger.exception(
                        f"Error processing article '{article.title}' "
                        f"for story summary",
                    )
                    continue

        if not story_summary:
            return "Story summary could not be generated."

        # Validate and fix the final summary
        prompt_context = (
            "A cumulative story summary. ONLY the summary text, "
            "no preambles or meta-commentary."
        )
        story_summary = self.judge_agent.validate_and_fix(
            story_summary,
            prompt_context,
        )

        # Validate named entities against source articles
        source_contents = self._get_article_contents(story.articles)
        return self.name_validation_agent.validate_and_fix(
            story_summary,
            source_contents,
        )

    def _refine_story_summary(
        self,
        story_summary: str,
        story: "Story",
    ) -> str:
        """
        Refine and polish the story summary.

        Args:
            story_summary: The raw story summary to refine
            story: Story object with source articles for validation

        Returns:
            Refined story summary

        """
        try:
            prompt_template = get_prompt(
                self.provider_name,
                "story_summary_refinement.txt",
            )
            prompt = prompt_template.format(story_summary=story_summary)

            refined_summary = self.provider.generate(
                prompt,
                {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )

            # Validate and fix if needed
            prompt_context = (
                "A refined 2-3 sentence story summary. ONLY the summary, "
                "no preambles or bullet points."
            )
            refined_summary = self.judge_agent.validate_and_fix(
                refined_summary,
                prompt_context,
            )

            # Validate named entities against source articles
            source_contents = self._get_article_contents(story.articles)
            refined_summary = self.name_validation_agent.validate_and_fix(
                refined_summary,
                source_contents,
            )

            logger.debug("Refined story summary")

        except Exception:
            logger.exception("Error refining story summary")
            return story_summary

        else:
            return refined_summary

    def _extract_additional_points(
        self,
        article: Article,
        story_summary: str,
    ) -> str:
        """
        Extract additional interesting points from an article.

        Args:
            article: Article to analyze
            story_summary: The cumulative story summary

        Returns:
            Additional points or empty string if no new information

        """
        if not article.content:
            return "Content not available."

        try:
            content = str(
                article.content[:MAX_ARTICLE_CONTENT_LENGTH]
                if len(article.content) > MAX_ARTICLE_CONTENT_LENGTH
                else article.content,
            )

            prompt_template = get_prompt(
                self.provider_name,
                "additional_points_extraction.txt",
            )
            prompt = prompt_template.format(
                story_summary=story_summary,
                title=article.title,
                content=content,
            )

            additional_points = self.provider.generate(
                prompt,
                {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )

            # Validate and fix if needed. Allow empty since the prompt
            # says to return nothing if no significant points.
            prompt_context = (
                "1-2 bullet points of additional information, or nothing. "
                "ONLY bullet points, no preambles or explanations."
            )
            additional_points = self.judge_agent.validate_and_fix(
                additional_points,
                prompt_context,
                allow_empty=True,
            )

            # Validate named entities against source article
            if additional_points:
                additional_points = (
                    self.name_validation_agent.validate_and_fix(
                        additional_points,
                        [article.content],
                    )
                )

            logger.debug(
                f"Extracted additional points from: {article.title[:50]}...",
            )

        except Exception:
            logger.exception(
                f"Error extracting additional points from '{article.title}'",
            )
            return (
                article.content[:MAX_ARTICLE_LENGTH_FALLBACK] + "..."
                if len(article.content) > MAX_ARTICLE_LENGTH_FALLBACK
                else article.content
            )

        else:
            return additional_points

    def _summarize_source_additional_points(
        self,
        source: str,
        additional_points_list: list[str],
        story_summary: str,
        source_articles: list[Article],
    ) -> str:
        """
        Summarize additional points from articles of a single source.

        Args:
            source: Name of the news source
            additional_points_list: List of additional points extracted
                from multiple articles of this source
            story_summary: The cumulative story summary
            source_articles: List of articles from this source for
                name validation

        Returns:
            Consolidated summary of additional points for this source

        """
        # Filter out empty or invalid points
        valid_points = [
            point.strip()
            for point in additional_points_list
            if point
            and point.strip()
            and point.strip() != "Content not available."
        ]

        if not valid_points:
            return ""

        try:
            # Format additional points list for the prompt
            points_text = "\n".join(
                f"- {point}" for point in valid_points if point
            )

            prompt_template = get_prompt(
                self.provider_name,
                "source_additional_points_summarization.txt",
            )
            prompt = prompt_template.format(
                story_summary=story_summary,
                source_name=source,
                additional_points_list=points_text,
            )

            source_summary = self.provider.generate(
                prompt,
                {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )

            # Validate and fix if needed. Allow empty since the prompt
            # says to return nothing if no significant points.
            prompt_context = (
                "1-2 bullet points (maximum 3 in exceptional cases) "
                "consolidating additional information from a source, or "
                "nothing. "
                "ONLY bullet points, no preambles, meta-commentary, or "
                "explanatory text. Do not include phrases like 'here are', "
                "'summarizing', 'this source provides', or similar."
            )
            source_summary = self.judge_agent.validate_and_fix(
                source_summary,
                prompt_context,
                allow_empty=True,
            )

            # Validate named entities against source articles
            if source_summary:
                source_contents = self._get_article_contents(source_articles)
                source_summary = self.name_validation_agent.validate_and_fix(
                    source_summary,
                    source_contents,
                )

            logger.debug(
                f"Summarized additional points for source: {source}",
            )

        except Exception:
            logger.exception(
                f"Error summarizing additional points for source '{source}'",
            )
            # Fallback: return first valid point if available
            return valid_points[0] if valid_points else ""

        else:
            return source_summary

    def summarize_articles_batch(
        self,
        articles: list[Article],
    ) -> dict[str, str]:
        """
        Summarize multiple articles.

        Args:
            articles: list of Article objects

        Returns:
            dictionary mapping article URLs to summaries

        """
        logger.info(f"Summarizing {len(articles)} articles")
        summaries = {}

        for i, article in enumerate(articles):
            try:
                summary = self.summarize_article(article)
                summaries[article.url] = summary
                article.summary = summary

                if (i + 1) % 10 == 0:
                    logger.info(f"Summarized {i + 1}/{len(articles)} articles")

            except Exception:
                logger.exception("Error in batch summarization")
                summaries[article.url] = article.content[
                    :MAX_ARTICLE_LENGTH_FALLBACK
                ]

        logger.info(f"Completed summarization of {len(summaries)} articles")
        return summaries

    def _sort_articles(self, articles: list[Article]) -> list[Article]:
        """
        Order articles based on configuration.

        Args:
            articles: List of articles to order

        Returns:
            Ordered list of articles

        """
        if self.article_order == "chronological":
            return sorted(articles, key=lambda a: a.published_date)
        if self.article_order == "source":
            return sorted(articles, key=lambda a: a.source)
        logger.warning(
            f"Unknown article order '{self.article_order}', "
            f"using original order",
        )
        return articles

    def _create_story_summary_prompt(
        self,
        article: Article,
        existing_summary: str,
    ) -> str:
        """
        Create prompt for building story summary.

        Args:
            article: Article to incorporate into summary
            existing_summary: Current cumulative summary

        Returns:
            Formatted prompt string

        """
        content = (
            article.content[:MAX_ARTICLE_CONTENT_LENGTH]
            if len(article.content) > MAX_ARTICLE_CONTENT_LENGTH
            else article.content
        )

        if existing_summary:
            existing_text = f"Current Story Summary:\n{existing_summary}\n"
        else:
            existing_text = ""

        prompt_template = get_prompt(
            self.provider_name,
            "story_summary_building.txt",
        )
        return prompt_template.format(
            existing_summary=existing_text,
            title=article.title,
            content=content,
        )

    def _create_summarization_prompt(self, article: Article) -> str:
        """
        Create prompt for article summarization.

        Args:
            article: Article to summarize

        Returns:
            Formatted prompt string

        """
        content = (
            article.content[:MAX_ARTICLE_CONTENT_LENGTH]
            if len(article.content) > MAX_ARTICLE_CONTENT_LENGTH
            else article.content
        )

        prompt_template = get_prompt(
            self.provider_name,
            "article_summarization.txt",
        )
        return prompt_template.format(
            title=article.title,
            source=article.source,
            content=content,
        )

    def _get_article_contents(self, articles: list[Article]) -> list[str]:
        """
        Extract content strings from articles for name validation.

        Args:
            articles: List of Article objects

        Returns:
            List of article content strings (including titles)

        """
        contents = []
        for article in articles:
            # Include title and content for name matching
            if article.title:
                contents.append(article.title)
            if article.content:
                contents.append(article.content)
        return contents
