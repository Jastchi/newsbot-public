"""Tests for Report Generator Agent"""

from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock, mock_open, patch

from newsbot.agents.report_agent import ReportGeneratorAgent
from newsbot.models import SentimentResult
from utilities import models as config_models


class TestReportGeneratorAgent:
    """Test cases for ReportGeneratorAgent"""

    def test_init(self, sample_config):
        """Test agent initialization"""
        agent = ReportGeneratorAgent(sample_config)

        assert agent.config == sample_config
        assert agent.format == "html"
        assert agent.include_summaries is True

    def test_init_with_defaults(self):
        """Test initialization with minimal config"""
        config = config_models.ConfigModel()
        agent = ReportGeneratorAgent(config)

        assert agent.format == "html"
        assert agent.include_summaries is True

    @patch("newsbot.agents.report_agent.ReportGeneratorAgent._save_report")
    def test_generate_top_stories_report(
        self, mock_save, sample_config, sample_articles
    ):
        """Test top stories report generation"""
        from newsbot.models import StoryAnalysis

        story_analyses: list[StoryAnalysis] = cast(
            list[StoryAnalysis],
            [
                {
                    "story": Mock(
                        story_id="story_1",
                        title="Test Story",
                        sources=["Source1", "Source2"],
                        article_count=2,
                        articles=sample_articles[:2],
                        earliest_date=datetime.now(),
                        latest_date=datetime.now(),
                        source_additional_points=None,
                    ),
                    "source_summaries": {
                        "Source1": [
                            {
                                "article": sample_articles[0],
                                "title": "Article 1",
                                "summary": "Summary 1",
                            }
                        ],
                        "Source2": [
                            {
                                "article": sample_articles[1],
                                "title": "Article 2",
                                "summary": "Summary 2",
                            }
                        ],
                    },
                    "source_sentiments": {
                        "Source1": {
                            "polarity": 0.5,
                            "label": "positive",
                            "avg_sentiment": 0.5,
                        },
                        "Source2": {
                            "polarity": 0.3,
                            "label": "positive",
                            "avg_sentiment": 0.3,
                        },
                    },
                },
            ],
        )

        # Mock _save_report to return a Path
        mock_save.return_value = Path(
            "reports/test/news_report_20250109_120000.html"
        )

        agent = ReportGeneratorAgent(sample_config)
        report, report_path = agent.generate_top_stories_report(story_analyses)

        assert isinstance(report, str)
        assert len(report) > 0
        assert isinstance(report_path, str)
        assert report_path.endswith(".html")
        # _save_report is called twice: once for main report, once for email report
        assert mock_save.call_count == 2

    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_report(self, mock_makedirs, mock_file, sample_config):
        """Test report saving"""
        agent = ReportGeneratorAgent(sample_config)
        report_content = "<html><body>Test Report</body></html>"

        report_path = agent._save_report(
            report_content, folder="reports/TestConfig"
        )

        assert isinstance(report_path, Path)
        assert "news_report_" in str(report_path)
        assert str(report_path).endswith(".html")
        # File may be opened multiple times (for writing and for Supabase upload)
        assert mock_file.call_count >= 1
        # Check that write was called with the report content
        write_calls = [call for call in mock_file().write.call_args_list if call[0][0] == report_content]
        assert len(write_calls) >= 1

    def test_generate_html_report_includes_all_sections(
        self, sample_config, sample_articles
    ):
        """Test that HTML report includes all expected sections"""
        for article in sample_articles:
            article.sentiment = SentimentResult(
                article_url=article.url,
                source=article.source,
                polarity=0.5,
                subjectivity=0.6,
                compound=0.5,
                label="positive",
            )
            article.summary = "Test summary"

        from newsbot.models import StoryAnalysis

        sentiment_comparison: StoryAnalysis = cast(
            StoryAnalysis,
            {
                "by_source": {
                    "Test": {
                        "avg_sentiment": 0.5,
                        "positive_count": 1,
                        "negative_count": 0,
                        "neutral_count": 0,
                    }
                },
                "overall_avg": 0.5,
            },
        )

        agent = ReportGeneratorAgent(sample_config)
        report = agent._generate_html_report(
            articles=sample_articles,
            sentiment_comparison=sentiment_comparison,
            topics_sentiment={"AI": {"avg_sentiment": 0.5}},
            weekly_overview="Overview",
            key_topics=["AI"],
        )

        # Check for key sections
        assert "overview" in report.lower() or "summary" in report.lower()
        assert isinstance(report, str)

    def test_generate_markdown_top_stories_report(
        self, sample_config, sample_articles
    ):
        """Test markdown format for top stories report"""
        from utilities.models import ReportConfigModel
        config = sample_config.model_copy(
            update={
                "report": sample_config.report.model_copy(update={"format": "markdown"})
            }
        )

        from newsbot.models import StoryAnalysis

        story_analyses: list[StoryAnalysis] = cast(
            list[StoryAnalysis],
            [
                    {
                        "story": Mock(
                            story_id="story_1",
                            title="Test Story",
                            sources=["Source1"],
                            article_count=1,
                            articles=sample_articles[:1],
                            earliest_date=datetime.now(),
                            latest_date=datetime.now(),
                            source_additional_points=None,
                        ),
                    "source_summaries": {
                        "Source1": [
                            {
                                "article": sample_articles[0],
                                "title": "Article",
                                "summary": "Summary",
                            }
                        ]
                    },
                    "source_sentiments": {
                        "Source1": {
                            "polarity": 0.5,
                            "label": "positive",
                            "avg_sentiment": 0.5,
                        }
                    },
                },
            ],
        )

        agent = ReportGeneratorAgent(config)
        report = agent._generate_markdown_top_stories_report(story_analyses)

        assert isinstance(report, str)
        assert "#" in report or "*" in report  # Markdown formatting

    def test_generate_text_top_stories_report(
        self, sample_config, sample_articles
    ):
        """Test plain text format for top stories report"""
        from utilities.models import ReportConfigModel
        config = sample_config.model_copy(
            update={
                "report": sample_config.report.model_copy(update={"format": "text"})
            }
        )

        from newsbot.models import StoryAnalysis

        story_analyses: list[StoryAnalysis] = cast(
            list[StoryAnalysis],
            [
                    {
                        "story": Mock(
                            story_id="story_1",
                            title="Test Story",
                            sources=["Source1"],
                            article_count=1,
                            articles=sample_articles[:1],
                            earliest_date=datetime.now(),
                            latest_date=datetime.now(),
                            source_additional_points=None,
                        ),
                    "source_summaries": {
                        "Source1": [
                            {
                                "article": sample_articles[0],
                                "title": "Article",
                                "summary": "Summary",
                            }
                        ]
                    },
                    "source_sentiments": {
                        "Source1": {
                            "polarity": 0.5,
                            "label": "positive",
                            "avg_sentiment": 0.5,
                        }
                    },
                },
            ],
        )

        agent = ReportGeneratorAgent(config)
        report = agent._generate_text_top_stories_report(story_analyses)

        assert isinstance(report, str)
        assert "Test Story" in report

    @patch("pathlib.Path.open", side_effect=PermissionError("Access denied"))
    @patch("os.makedirs")
    def test_save_report_permission_error(
        self, mock_makedirs, mock_file, sample_config
    ):
        """Test handling of permission error when saving report"""
        agent = ReportGeneratorAgent(sample_config)

        with patch("newsbot.agents.report_agent.logger") as mock_logger:
            report_path = agent._save_report(
                "Test content", folder="reports/TestConfig"
            )
            # Should log exception and return empty Path
            assert mock_logger.exception.called
            assert report_path == Path()

    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_report_filename_format(
        self, mock_makedirs, mock_file, sample_config
    ):
        """Test that report filename has correct format"""
        agent = ReportGeneratorAgent(sample_config)
        report_path = agent._save_report(
            "Test content", folder="reports/TestConfig"
        )

        # Check that the returned path contains timestamp
        assert isinstance(report_path, Path)
        assert "news_report_" in report_path.name
        assert report_path.name.endswith(".html")
