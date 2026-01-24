"""
Report Generator Agent.

Generates news analysis reports
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from newsbot.constants import TZ
from newsbot.models import Article, StoryAnalysis
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)


class ReportGeneratorAgent:
    """Agent responsible for generating reports."""

    def __init__(self, config: ConfigModel) -> None:
        """
        Initialize the Report Generator Agent.

        Args:
            config: Configuration dictionary

        """
        self.config = config
        self.report_config = config.report

        self.format = self.report_config.format
        self.include_summaries = self.report_config.include_summaries

        # Set up Jinja2 template environment
        # Get absolute path to templates directory
        current_dir = Path(__file__).resolve().parent
        template_dir = current_dir.parent / "templates"

        if not template_dir.exists():
            logger.error(f"Template directory not found at {template_dir}")

        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True,
        )

    def generate_top_stories_report(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> tuple[str, str]:
        """
        Generate report focused on top stories.

        Args:
            story_analyses: list of dictionaries containing story,
                summaries, and sentiments

        Returns:
            Tuple of (report content as string, path to saved file)

        """
        logger.info("Generating top stories report")

        if self.format == "html":
            report = self._generate_html_top_stories_report(story_analyses)
        elif self.format == "markdown":
            report = self._generate_markdown_top_stories_report(story_analyses)
        else:  # text
            report = self._generate_text_top_stories_report(story_analyses)

        # Save report to file
        report_path = self._save_report(
            report,
            folder=f"reports/{self.config.name}",
        )

        logger.info("Top stories report generated successfully")

        self._generate_html_email_top_stories_report(
            story_analyses,
            report_path.name,
        )

        logger.info("Top stories email report generated successfully")

        return report, str(report_path)

    def _generate_html_report(
        self,
        articles: list[Article],
        sentiment_comparison: StoryAnalysis,
        topics_sentiment: dict[str, dict],
        weekly_overview: str,
        key_topics: list[str],
    ) -> str:
        """Generate HTML report."""
        # Group articles by source
        articles_by_source = {}
        for article in articles:
            if article.source not in articles_by_source:
                articles_by_source[article.source] = []
            articles_by_source[article.source].append(article)

        # Load template from file
        template = self.jinja_env.get_template("weekly_report.html")

        # Prepare data for template
        html = template.render(
            date=datetime.now(TZ).strftime("%B %d, %Y"),
            weekly_overview=weekly_overview,
            key_topics=key_topics,
            total_articles=len(articles),
            sources_count=len(articles_by_source),
            overall_sentiment=(
                f"{sentiment_comparison.get('overall_avg', 0):.3f}"
            ),
            sentiment_by_source=sentiment_comparison.get("by_source", {}),
            significant_differences=sentiment_comparison.get(
                "significant_differences",
                [],
            ),
            articles_by_source=articles_by_source,
            include_summaries=self.include_summaries,
        )

        logger.debug(topics_sentiment)

        return html

    def _generate_html_top_stories_report(
        self,
        story_analyses: list[StoryAnalysis],
        template_file: str = "top_stories_report.html",
    ) -> str:
        """Generate HTML report for top stories."""
        # Load template from file
        template = self.jinja_env.get_template(template_file)

        # Collect all unique sources
        all_sources = set()
        total_articles = 0
        for analysis in story_analyses:
            all_sources.update(analysis["story"].sources)
            total_articles += analysis["story"].article_count

        now = datetime.now(TZ)
        to_date = now.strftime("%d %B")
        from_date = (
            now
            - timedelta(
                days=self.config.scheduler.weekly_analysis.lookback_days,
            )
        ).strftime("%d %B")

        return template.render(
            topic=self.config.name or "Top Stories",
            story_analyses=story_analyses,
            story_count=len(story_analyses),
            total_articles=total_articles,
            sources=sorted(all_sources),
            date=now.strftime("%Y-%m-%d %H:%M:%S"),
            date_range=(f"{from_date} - {to_date}"),
        )

    def _generate_markdown_top_stories_report(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> str:
        """Generate Markdown report for top stories."""
        # Load template from file
        template = self.jinja_env.get_template("top_stories_report.md")

        # Collect all unique sources
        all_sources = set()
        total_articles = 0
        for analysis in story_analyses:
            all_sources.update(analysis["story"].sources)
            total_articles += analysis["story"].article_count

        return template.render(
            story_analyses=story_analyses,
            story_count=len(story_analyses),
            total_articles=total_articles,
            sources=sorted(all_sources),
            date=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _generate_text_top_stories_report(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> str:
        """Generate plain text report for top stories."""
        # Load template from file
        template = self.jinja_env.get_template("top_stories_report.txt")

        # Collect all unique sources
        all_sources = set()
        total_articles = 0
        for analysis in story_analyses:
            all_sources.update(analysis["story"].sources)
            total_articles += analysis["story"].article_count

        return template.render(
            story_analyses=story_analyses,
            story_count=len(story_analyses),
            total_articles=total_articles,
            sources=sorted(all_sources),
            date=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _generate_html_email_top_stories_report(
        self,
        story_analyses: list[StoryAnalysis],
        report_filename: str,
    ) -> None:
        """Generate HTML email report for top stories."""
        report = self._generate_html_top_stories_report(
            story_analyses,
            template_file="top_stories_report_email.html",
        )
        self._save_report(
            report,
            folder=f"reports/{self.config.name}/email_reports",
            filename=report_filename,
        )

    def _save_report(
        self,
        report: str,
        folder: str,
        filename: str | None = None,
    ) -> Path:
        """
        Save report to file.

        Returns:
            Path to the saved report file

        """
        try:
            # Create reports directory if it doesn't exist
            Path(folder).mkdir(parents=True, exist_ok=True)

            if not filename:
                # Generate filename
                timestamp = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
                ext = (
                    "html"
                    if self.format == "html"
                    else ("md" if self.format == "markdown" else "txt")
                )
                filename = f"news_report_{timestamp}.{ext}"

            filepath = Path(folder) / filename

            # Save report locally
            with filepath.open("w", encoding="utf-8") as f:
                f.write(report)


            logger.info(f"Report saved to {filename}")

        except Exception:
            logger.exception("Error saving report")
            return Path()
        else:
            return filepath

