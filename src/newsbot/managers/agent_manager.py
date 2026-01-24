"""Agent Manager for lazy initialization of pipeline agents."""

import logging

from newsbot.agents.report_agent import ReportGeneratorAgent
from newsbot.agents.scraper_agent import NewsScraperAgent
from newsbot.agents.sentiment_agent import SentimentAnalysisAgent
from newsbot.agents.story_clustering_agent import StoryClusteringAgent
from newsbot.agents.summarization_agent import SummarizationAgent
from newsbot.managers.database_manager import DatabaseManager
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages lazy initialization and lifecycle of pipeline agents."""

    def __init__(
        self,
        config: ConfigModel,
        database_manager: DatabaseManager | None = None,
    ) -> None:
        """
        Initialize the Agent Manager.

        Args:
            config: Configuration model for agent initialization
            database_manager: Optional database manager for URL checking
                optimization in scraper

        """
        self.config = config
        self._database_manager = database_manager
        self._scraper: NewsScraperAgent | None = None
        self._story_clustering: StoryClusteringAgent | None = None
        self._summarizer: SummarizationAgent | None = None
        self._sentiment_analyzer: SentimentAnalysisAgent | None = None
        self._report_generator: ReportGeneratorAgent | None = None

    @property
    def scraper(self) -> NewsScraperAgent:
        """Get the scraper agent, initializing if needed."""
        if self._scraper is None:
            logger.info("Initializing scraper agent...")
            # Pass URL check callback if database manager is available
            url_check = None
            if self._database_manager is not None:
                url_check = self._database_manager.url_exists
            self._scraper = NewsScraperAgent(self.config, url_check=url_check)
            logger.info("Scraper agent initialized")
        return self._scraper

    @property
    def story_clustering(self) -> StoryClusteringAgent:
        """Get the story clustering agent, initializing if needed."""
        if self._story_clustering is None:
            logger.info("Initializing story clustering agent...")
            self._story_clustering = StoryClusteringAgent(self.config)
            logger.info("Story clustering agent initialized")
        return self._story_clustering

    @property
    def summarizer(self) -> SummarizationAgent:
        """Get the summarization agent, initializing if needed."""
        if self._summarizer is None:
            logger.info("Initializing summarization agent...")
            self._summarizer = SummarizationAgent(self.config)
            logger.info("Summarization agent initialized")
        return self._summarizer

    @property
    def sentiment_analyzer(self) -> SentimentAnalysisAgent:
        """Get the sentiment analysis agent, initializing if needed."""
        if self._sentiment_analyzer is None:
            logger.info("Initializing sentiment analysis agent...")
            self._sentiment_analyzer = SentimentAnalysisAgent(self.config)
            logger.info("Sentiment analysis agent initialized")
        return self._sentiment_analyzer

    @property
    def report_generator(self) -> ReportGeneratorAgent:
        """Get the report generator agent, initializing if needed."""
        if self._report_generator is None:
            logger.info("Initializing report generator agent...")
            self._report_generator = ReportGeneratorAgent(self.config)
            logger.info("Report generator agent initialized")
        return self._report_generator

    def get_agent_status(self) -> dict[str, str]:
        """
        Get initialization status for all agents.

        Returns:
            Dictionary mapping agent names to their status

        """
        return {
            "scraper": "initialized"
            if self._scraper is not None
            else "not_initialized",
            "story_clustering": "initialized"
            if self._story_clustering is not None
            else "not_initialized",
            "summarizer": "initialized"
            if self._summarizer is not None
            else "not_initialized",
            "sentiment_analyzer": "initialized"
            if self._sentiment_analyzer is not None
            else "not_initialized",
            "report_generator": "initialized"
            if self._report_generator is not None
            else "not_initialized",
        }
