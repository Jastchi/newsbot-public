"""
Report Generator Agent.

Generates news analysis reports
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from newsbot.color_utils import derive_color_palette, derive_dark_palette
from newsbot.constants import TZ
from newsbot.models import Article, StoryAnalysis
from newsbot.preview_fixtures import build_preview_story_analyses
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
        articles_by_source: dict[str, list[Article]] = defaultdict(list)
        for article in articles:
            articles_by_source[article.source].append(article)

        template = self.jinja_env.get_template("weekly_report.html")

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

        palette = derive_color_palette(
            self.config.hero_color_primary,
            self.config.hero_color_secondary,
            self.config.hero_color_middle,
        )
        return template.render(
            topic=self.config.name or "Top Stories",
            story_analyses=story_analyses,
            story_count=len(story_analyses),
            total_articles=total_articles,
            sources=sorted(all_sources),
            date=now.strftime("%Y-%m-%d %H:%M:%S"),
            date_range=(f"{from_date} - {to_date}"),
            **palette,
        )

    def _render_simple_top_stories_template(
        self,
        story_analyses: list[StoryAnalysis],
        template_file: str,
    ) -> str:
        """Render a markdown or text top-stories template."""
        template = self.jinja_env.get_template(template_file)
        all_sources: set[str] = set()
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

    def _generate_markdown_top_stories_report(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> str:
        """Generate Markdown report for top stories."""
        return self._render_simple_top_stories_template(
            story_analyses, "top_stories_report.md",
        )

    def _generate_text_top_stories_report(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> str:
        """Generate plain text report for top stories."""
        return self._render_simple_top_stories_template(
            story_analyses, "top_stories_report.txt",
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

    @staticmethod
    def render_preview(
        topic: str,
        primary: str,
        secondary: str,
        middle: str | None = None,
    ) -> str:
        """
        Render the email template with mock data for admin preview.

        Args:
            topic: The newsletter topic name.
            primary: Primary brand hex color.
            secondary: Secondary brand hex color.
            middle: Optional middle brand hex color for the gradient.

        Returns:
            Rendered HTML string.

        """
        current_dir = Path(__file__).resolve().parent
        template_dir = current_dir.parent / "templates"
        loader = FileSystemLoader(template_dir)
        env = Environment(loader=loader, autoescape=True)
        template = env.get_template("top_stories_report_email.html")

        now = datetime.now(TZ)
        analyses = build_preview_story_analyses()

        all_sources: set[str] = set()
        total_articles = 0
        for analysis in analyses:
            all_sources.update(analysis["story"].sources)
            total_articles += analysis["story"].article_count

        from_date = (now - timedelta(days=7)).strftime("%d %B")
        to_date = now.strftime("%d %B")

        palette = derive_color_palette(primary, secondary, middle)
        html = template.render(
            topic=topic,
            story_analyses=analyses,
            story_count=len(analyses),
            total_articles=total_articles,
            sources=sorted(all_sources),
            date=now.strftime("%Y-%m-%d %H:%M:%S"),
            date_range=f"{from_date} - {to_date}",
            **palette,
        )
        return ReportGeneratorAgent._inject_dark_mode_toggle(
            html,
            primary,
            secondary,
        )

    @staticmethod
    def _inject_dark_mode_toggle(
        html: str,
        primary: str,
        secondary: str,
    ) -> str:
        """Inject dark-mode toggle and CSS into preview HTML."""
        dk = derive_dark_palette(primary, secondary)

        dark_css = f"""
<style id="preview-dark-overrides">
body[data-theme="dark"] {{
    background: {dk['dk_bg']} !important;
    color: {dk['dk_text']} !important;
}}
body[data-theme="dark"] .intro {{
    box-shadow: 0 2px 12px {dk['dk_shadow']} !important;
}}
body[data-theme="dark"] .global-summary,
body[data-theme="dark"] .story-summary,
body[data-theme="dark"] .footer {{
    background: {dk['dk_tint']} !important;
    border-color: {dk['dk_border']} !important;
    color: {dk['dk_text']} !important;
}}
body[data-theme="dark"] .story {{
    background: {dk['dk_card']} !important;
    box-shadow: 0 1px 4px {dk['dk_story_shadow']} !important;
}}
body[data-theme="dark"] h2,
body[data-theme="dark"] h3 {{
    color: {dk['dk_primary']} !important;
    border-color: {dk['dk_border']} !important;
}}
body[data-theme="dark"] h3.sources-header,
body[data-theme="dark"] .stories-list-header {{
    color: {dk['dk_muted']} !important;
}}
body[data-theme="dark"] .story-title {{
    color: {dk['dk_primary']} !important;
    border-top-color: {dk['dk_primary']} !important;
}}
body[data-theme="dark"] .story-summary .story-summary-title {{
    color: {dk['dk_primary']} !important;
}}
body[data-theme="dark"] .story-summary-label,
body[data-theme="dark"] .stories-list-item-num {{
    color: {dk['dk_accent']} !important;
}}
body[data-theme="dark"] .story-summary-content,
body[data-theme="dark"] .additional-insights {{
    color: {dk['dk_text']} !important;
}}
body[data-theme="dark"] .additional-insights strong {{
    color: {dk['dk_primary']} !important;
}}
body[data-theme="dark"] .source-coverage {{
    border-left-color: {dk['dk_border']} !important;
}}
body[data-theme="dark"] .source-coverage > table td {{
    color: {dk['dk_text']} !important;
    border-bottom-color: {dk['dk_border']} !important;
}}
body[data-theme="dark"] .footer-divider {{
    border-top-color: {dk['dk_border']} !important;
}}
body[data-theme="dark"] .footer {{
    color: {dk['dk_text_muted']} !important;
}}
body[data-theme="dark"] a {{
    color: {dk['dk_primary']} !important;
}}
body[data-theme="dark"] .stories-list-item {{
    border-bottom-color: {dk['dk_border']} !important;
}}
body[data-theme="dark"] .stories-list-item a {{
    color: {dk['dk_text_muted']} !important;
}}
body[data-theme="dark"] .stories-list-item a:hover {{
    color: {dk['dk_primary']} !important;
}}
body[data-theme="dark"] .story-meta,
body[data-theme="dark"] .summary,
body[data-theme="dark"] .article-title {{
    color: {dk['dk_text']} !important;
}}
</style>"""

        toggle = """
<div id="preview-dark-toggle" style="
    position:fixed; top:12px; right:12px; z-index:9999;
    background:rgba(0,0,0,0.55); border-radius:20px;
    padding:5px 14px; cursor:pointer; font-size:13px;
    color:#fff; font-family:sans-serif;
    backdrop-filter:blur(6px); user-select:none;
    border:1px solid rgba(255,255,255,0.15);
    transition:background 0.2s;">&#9790; Dark</div>
<script>
(function () {
    var btn = document.getElementById("preview-dark-toggle");
    btn.addEventListener("click", function () {
        var isDark = document.body.getAttribute("data-theme") === "dark";
        document.body.setAttribute("data-theme", isDark ? "light" : "dark");
        btn.innerHTML = isDark ? "&#9790; Dark" : "&#9728; Light";
    });
})();
</script>"""

        return (
            html.replace("</head>", dark_css + "\n</head>", 1)
            .replace("</body>", toggle + "\n</body>", 1)
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

