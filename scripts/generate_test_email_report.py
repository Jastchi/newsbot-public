"""
Script to generate a test email report using sample data.

This script creates sample news stories and generates an email report
using the top_stories_report_email.html template.

Run with arguments --email --email-receivers to send the report via
email.
"""

import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from after_analysis.email_sender import execute as send_email
from newsbot.agents.report_agent import ReportGeneratorAgent
from newsbot.constants import TZ
from newsbot.models import (
    AnalysisData,
    Article,
    SentimentResult,
    SentimentSummary,
    Story,
    StoryAnalysis,
    SummaryItem,
)
from utilities import models as config_models
from utilities.django_models import Article as DjangoArticle

# Sentiment thresholds for classification
POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1


def create_sample_articles() -> list[Article]:
    """
    Create sample articles for testing.

    Multiple articles per source are included.

    Returns:
        List of Article objects with sample data.

    """
    now = datetime.now(TZ)
    base_time = now - timedelta(days=2)

    return [
        # TechNews articles (3 articles)
        Article(
            title="Major Tech Company Announces Revolutionary AI Breakthrough",
            content=(
                "A leading technology company has unveiled a groundbreaking "
                "artificial intelligence system that promises to transform "
                "multiple industries. The new AI model demonstrates "
                "unprecedented capabilities in natural language understanding "
                "and problem-solving."
            ),
            source="TechNews",
            url="https://technews.example.com/ai-breakthrough",
            published_date=base_time + timedelta(hours=2),
            scraped_date=base_time + timedelta(hours=2, minutes=15),
            summary=(
                "Tech company reveals advanced AI system with superior "
                "language understanding capabilities."
            ),
            sentiment=SentimentResult(
                article_url="https://technews.example.com/ai-breakthrough",
                source="TechNews",
                polarity=0.6,
                subjectivity=0.4,
                compound=0.7,
                label="positive",
            ),
        ),
        Article(
            title="Tech Industry Leaders Discuss AI Regulation Framework",
            content=(
                "Top executives from major technology companies gathered to "
                "discuss the need for comprehensive AI regulation. The "
                "consensus was that self-regulation may not be sufficient."
            ),
            source="TechNews",
            url="https://technews.example.com/ai-regulation",
            published_date=base_time + timedelta(hours=8),
            scraped_date=base_time + timedelta(hours=8, minutes=30),
            summary=(
                "Tech leaders call for government involvement in AI "
                "regulation standards."
            ),
            sentiment=SentimentResult(
                article_url="https://technews.example.com/ai-regulation",
                source="TechNews",
                polarity=0.2,
                subjectivity=0.5,
                compound=0.3,
                label="positive",
            ),
        ),
        Article(
            title="New AI Chip Architecture Promises 10x Performance Boost",
            content=(
                "Engineers have developed a new chip architecture "
                "specifically designed for AI workloads. Early benchmarks "
                "show dramatic "
                "performance improvements over existing solutions."
            ),
            source="TechNews",
            url="https://technews.example.com/ai-chip",
            published_date=base_time + timedelta(hours=14),
            scraped_date=base_time + timedelta(hours=14, minutes=20),
            summary=(
                "Breakthrough chip design could revolutionize AI computing "
                "performance."
            ),
            sentiment=SentimentResult(
                article_url="https://technews.example.com/ai-chip",
                source="TechNews",
                polarity=0.5,
                subjectivity=0.3,
                compound=0.6,
                label="positive",
            ),
        ),
        # BusinessDaily articles (2 articles)
        Article(
            title="AI Development Raises Concerns About Job Displacement",
            content=(
                "The rapid advancement of artificial intelligence technology "
                "has sparked concerns among workers and labor advocates. "
                "Experts warn that automation could lead to significant job "
                "losses across various sectors."
            ),
            source="BusinessDaily",
            url="https://businessdaily.example.com/ai-jobs",
            published_date=base_time + timedelta(hours=4),
            scraped_date=base_time + timedelta(hours=4, minutes=20),
            summary=(
                "AI progress triggers worries about potential job losses "
                "and economic disruption."
            ),
            sentiment=SentimentResult(
                article_url="https://businessdaily.example.com/ai-jobs",
                source="BusinessDaily",
                polarity=-0.3,
                subjectivity=0.5,
                compound=-0.4,
                label="negative",
            ),
        ),
        Article(
            title=(
                "Economists Predict AI Will Create More Jobs Than It Destroys"
            ),
            content=(
                "A new economic study suggests that while AI will eliminate "
                "some jobs, it will create many more in emerging fields. "
                "The key is retraining the workforce for new opportunities."
            ),
            source="BusinessDaily",
            url="https://businessdaily.example.com/ai-jobs-creation",
            published_date=base_time + timedelta(hours=12),
            scraped_date=base_time + timedelta(hours=12, minutes=45),
            summary=(
                "Study finds net positive job creation from AI, with emphasis "
                "on worker retraining."
            ),
            sentiment=SentimentResult(
                article_url="https://businessdaily.example.com/ai-jobs-creation",
                source="BusinessDaily",
                polarity=0.1,
                subjectivity=0.6,
                compound=0.2,
                label="positive",
            ),
        ),
        # ScienceWeekly articles (2 articles)
        Article(
            title="New AI Technology Shows Promise in Medical Research",
            content=(
                "Researchers are exploring how artificial intelligence can "
                "accelerate medical discoveries. Early results suggest AI "
                "could help identify new treatments faster than traditional "
                "methods."
            ),
            source="ScienceWeekly",
            url="https://scienceweekly.example.com/ai-medicine",
            published_date=base_time + timedelta(hours=6),
            scraped_date=base_time + timedelta(hours=6, minutes=10),
            summary=(
                "AI demonstrates potential to speed up medical research "
                "and treatment development."
            ),
            sentiment=SentimentResult(
                article_url="https://scienceweekly.example.com/ai-medicine",
                source="ScienceWeekly",
                polarity=0.4,
                subjectivity=0.3,
                compound=0.5,
                label="positive",
            ),
        ),
        Article(
            title=(
                "AI-Assisted Drug Discovery Yields Promising Cancer Treatment"
            ),
            content=(
                "Scientists using AI algorithms have identified a new "
                "compound "
                "that shows remarkable effectiveness against certain cancer "
                "types. Clinical trials are set to begin next year."
            ),
            source="ScienceWeekly",
            url="https://scienceweekly.example.com/ai-cancer",
            published_date=base_time + timedelta(hours=16),
            scraped_date=base_time + timedelta(hours=16, minutes=25),
            summary=(
                "AI-identified cancer treatment compound enters clinical "
                "trial phase."
            ),
            sentiment=SentimentResult(
                article_url="https://scienceweekly.example.com/ai-cancer",
                source="ScienceWeekly",
                polarity=0.6,
                subjectivity=0.2,
                compound=0.7,
                label="positive",
            ),
        ),
        # WorldNews articles (2 articles)
        Article(
            title="Global Climate Summit Reaches Historic Agreement",
            content=(
                "World leaders have reached a landmark agreement on climate "
                "action at the latest international summit. The deal includes "
                "ambitious targets for reducing carbon emissions by 2030."
            ),
            source="WorldNews",
            url="https://worldnews.example.com/climate-summit",
            published_date=base_time + timedelta(days=1, hours=1),
            scraped_date=base_time + timedelta(days=1, hours=1, minutes=30),
            summary=(
                "International climate summit produces significant agreement "
                "on emission reduction targets."
            ),
            sentiment=SentimentResult(
                article_url="https://worldnews.example.com/climate-summit",
                source="WorldNews",
                polarity=0.5,
                subjectivity=0.4,
                compound=0.6,
                label="positive",
            ),
        ),
        Article(
            title="Climate Summit Participants Commit to Renewable Energy",
            content=(
                "Nations at the climate summit have pledged to increase "
                "renewable energy investments by 50% over the next five "
                "years. This represents the largest collective commitment "
                "to clean energy in history."
            ),
            source="WorldNews",
            url="https://worldnews.example.com/renewable-commitment",
            published_date=base_time + timedelta(days=1, hours=5),
            scraped_date=base_time + timedelta(days=1, hours=5, minutes=15),
            summary=(
                "Historic renewable energy investment commitment announced "
                "at climate summit."
            ),
            sentiment=SentimentResult(
                article_url="https://worldnews.example.com/renewable-commitment",
                source="WorldNews",
                polarity=0.6,
                subjectivity=0.3,
                compound=0.7,
                label="positive",
            ),
        ),
        # PolicyReview articles (2 articles)
        Article(
            title="Climate Agreement Faces Implementation Challenges",
            content=(
                "While the climate summit produced an agreement, experts "
                "question whether countries will follow through on their "
                "commitments. Past agreements have struggled with enforcement."
            ),
            source="PolicyReview",
            url="https://policyreview.example.com/climate-challenges",
            published_date=base_time + timedelta(days=1, hours=3),
            scraped_date=base_time + timedelta(days=1, hours=3, minutes=45),
            summary=(
                "Experts express skepticism about implementation of new "
                "climate agreement."
            ),
            sentiment=SentimentResult(
                article_url="https://policyreview.example.com/climate-challenges",
                source="PolicyReview",
                polarity=-0.2,
                subjectivity=0.6,
                compound=-0.3,
                label="neutral",
            ),
        ),
        Article(
            title="Analysis: Climate Agreement Enforcement Mechanisms",
            content=(
                "Policy experts analyze the enforcement mechanisms in the "
                "new climate agreement. While stronger than previous deals, "
                "some gaps remain in accountability measures."
            ),
            source="PolicyReview",
            url="https://policyreview.example.com/enforcement",
            published_date=base_time + timedelta(days=1, hours=7),
            scraped_date=base_time + timedelta(days=1, hours=7, minutes=30),
            summary=(
                "Policy analysis reveals both strengths and weaknesses in "
                "climate agreement enforcement."
            ),
            sentiment=SentimentResult(
                article_url="https://policyreview.example.com/enforcement",
                source="PolicyReview",
                polarity=-0.1,
                subjectivity=0.7,
                compound=-0.2,
                label="neutral",
            ),
        ),
    ]


def create_sample_stories(articles: list[Article]) -> list[Story]:
    """Create sample stories from articles."""
    # Group articles by topic
    # AI story: TechNews (3), BusinessDaily (2), ScienceWeekly (2)
    # = 7 articles total
    ai_articles = [
        articles[0],  # TechNews - AI Breakthrough
        articles[1],  # TechNews - AI Regulation
        articles[2],  # TechNews - AI Chip
        articles[3],  # BusinessDaily - Job Displacement
        articles[4],  # BusinessDaily - Job Creation
        articles[5],  # ScienceWeekly - Medical Research
        articles[6],  # ScienceWeekly - Cancer Treatment
    ]
    # Climate story: WorldNews (2), PolicyReview (2) = 4 articles
    climate_articles = [
        articles[7],  # WorldNews - Climate Summit
        articles[8],  # WorldNews - Renewable Energy
        articles[9],  # PolicyReview - Implementation Challenges
        articles[10],  # PolicyReview - Enforcement Mechanisms
    ]

    return [
        Story(
            story_id="story-1",
            title="AI Technology Breakthrough and Its Implications",
            articles=ai_articles,
            sources=["TechNews", "BusinessDaily", "ScienceWeekly"],
            article_count=len(ai_articles),
            earliest_date=min(a.published_date for a in ai_articles),
            latest_date=max(a.published_date for a in ai_articles),
            story_summary=(
                "Major developments in artificial intelligence technology "
                "have generated both excitement and concern. While the "
                "technology shows promise in fields like medicine, there "
                "are ongoing worries about its impact on employment."
            ),
            source_additional_points={
                "TechNews": (
                    "TechNews provides comprehensive coverage with three "
                    "articles covering the AI breakthrough, regulatory "
                    "discussions, and new chip architecture developments."
                ),
                "BusinessDaily": (
                    "BusinessDaily offers balanced perspective with articles "
                    "on job displacement concerns and economic opportunities "
                    "from AI advancement."
                ),
                "ScienceWeekly": (
                    "ScienceWeekly focuses on medical applications with "
                    "coverage of AI in research and promising cancer "
                    "treatment discoveries."
                ),
            },
        ),
        Story(
            story_id="story-2",
            title="Climate Summit Agreement and Implementation Concerns",
            articles=climate_articles,
            sources=["WorldNews", "PolicyReview"],
            article_count=len(climate_articles),
            earliest_date=min(a.published_date for a in climate_articles),
            latest_date=max(a.published_date for a in climate_articles),
            story_summary=(
                "World leaders reached a new climate agreement, but questions "
                "remain about whether countries will meet their commitments. "
                "The deal sets ambitious targets, but enforcement mechanisms "
                "are still being developed."
            ),
            source_additional_points={
                "WorldNews": (
                    "The agreement includes financial support for developing "
                    "countries to transition to clean energy."
                ),
                "PolicyReview": (
                    "Previous climate agreements have had mixed success, "
                    "with some countries failing to meet their targets."
                ),
            },
        ),
    ]


def create_story_analyses(stories: list[Story]) -> list[StoryAnalysis]:
    """Create story analyses with summaries and sentiments."""
    story_analyses = []

    for story in stories:
        # Group articles by source for summaries
        source_summaries: dict[str, list[SummaryItem]] = {}
        source_sentiments: dict[str, SentimentSummary] = {}

        for article in story.articles:
            # Add to source summaries
            if article.source not in source_summaries:
                source_summaries[article.source] = []
            source_summaries[article.source].append(
                SummaryItem(
                    article=article,
                    summary=article.summary,
                ),
            )

            # Calculate sentiment summaries per source
            if article.source not in source_sentiments:
                source_sentiments[article.source] = SentimentSummary(
                    avg_sentiment=0.0,
                    label="neutral",
                    article_count=0,
                    sentiments=[],
                )

            sentiment_summary = source_sentiments[article.source]
            if article.sentiment:
                sentiment_summary["sentiments"].append(article.sentiment)
                sentiment_summary["article_count"] += 1

        # Calculate average sentiments per source
        for sentiment_summary in source_sentiments.values():
            if sentiment_summary["article_count"] > 0:
                sentiments = sentiment_summary["sentiments"]
                avg_compound = sum(s.compound for s in sentiments) / len(
                    sentiments,
                )
                sentiment_summary["avg_sentiment"] = avg_compound

                # Determine label based on average
                if avg_compound > POSITIVE_THRESHOLD:
                    sentiment_summary["label"] = "positive"
                elif avg_compound < NEGATIVE_THRESHOLD:
                    sentiment_summary["label"] = "negative"
                else:
                    sentiment_summary["label"] = "neutral"

        story_analyses.append(
            StoryAnalysis(
                story=story,
                source_summaries=source_summaries,
                source_sentiments=source_sentiments,
            ),
        )

    return story_analyses


def load_from_database(days_back: int = 7) -> list[StoryAnalysis]:
    """
    Load articles from database and create story analyses.

    Note: This is a simplified version. In production, you would use
    the full pipeline (clustering, summarization, sentiment analysis).

    Args:
        days_back: Number of days to look back for articles

    Returns:
        List of story analyses (empty if database not available).

    """
    try:
        # Django is automatically set up when django_models is imported

        cutoff_date = datetime.now(TZ) - timedelta(days=days_back)

        db_articles = DjangoArticle.objects.filter(
            scraped_date__gte=cutoff_date,
        ).all()

        if not db_articles:
            print(f"No articles found in database (last {days_back} days)")
            print("Falling back to sample data...")
            return []

        # Convert to Article objects
        articles = [
            Article(
                title=db_art.title,
                content=db_art.content or "",
                source=db_art.source,
                url=db_art.url,
                published_date=db_art.published_date or datetime.now(TZ),
                scraped_date=db_art.scraped_date or datetime.now(TZ),
                summary=db_art.summary,
                sentiment=(
                    SentimentResult(
                        article_url=db_art.url,
                        source=db_art.source,
                        polarity=0.0,
                        subjectivity=0.0,
                        compound=db_art.sentiment_score or 0.0,
                        label=db_art.sentiment_label or "neutral",
                    )
                    if db_art.sentiment_score is not None
                    else None
                ),
            )
            for db_art in db_articles
        ]

        print(f"Loaded {len(articles)} articles from database")

        # Create simple stories (one per source for demo)
        # In production, you'd use the clustering agent
        stories = create_sample_stories(articles[:5])  # Limit for demo
        return create_story_analyses(stories)

    except Exception as e:
        print(f"Error loading from database: {e}")
        print("Falling back to sample data...")
        return []


def main() -> None:
    """Generate a test email report."""
    parser = ArgumentParser(description="Generate a test email report")
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Load articles from database instead of using sample data",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help=(
            "Number of days to look back when loading from database "
            "(default: 7)"
        ),
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send the report via email using email_sender",
    )
    parser.add_argument(
        "--email-receivers",
        nargs="*",
        default=None,
        help=(
            "Email addresses to send the report to (overrides database "
            "recipients). Use --email-receivers with no addresses to send "
            "only to sender."
        ),
    )
    args = parser.parse_args()

    if args.from_db:
        print("Loading data from database...")
        story_analyses = load_from_database(args.days_back)
        if not story_analyses:
            print("Creating sample data instead...")
            articles = create_sample_articles()
            stories = create_sample_stories(articles)
            story_analyses = create_story_analyses(stories)
    else:
        print("Creating sample data...")
        articles = create_sample_articles()
        stories = create_sample_stories(articles)
        story_analyses = create_story_analyses(stories)

    print(f"Created {len(story_analyses)} story analyses")

    # Create a minimal config for the report generator
    config = config_models.ConfigModel(
        name="Test Report",
        report=config_models.ReportConfigModel(
            format="html",
            include_summaries=True,
        ),
        scheduler=config_models.SchedulerConfigModel(
            weekly_analysis=config_models.WeeklyAnalysisConfigModel(
                lookback_days=7,
            ),
        ),
    )

    print("Generating email report...")
    report_agent = ReportGeneratorAgent(config)

    # Generate regular report first (email_sender expects this)
    _regular_report, regular_report_path = (
        report_agent.generate_top_stories_report(story_analyses)
    )

    print("Email report generated successfully!")
    print(f"Report saved to: {regular_report_path}")

    # Send email if requested
    if args.email:
        print("\nSending email...")
        try:

            # Calculate date range
            now = datetime.now(TZ)
            from_date = now - timedelta(
                days=config.scheduler.weekly_analysis.lookback_days,
            )
            to_date = now

            # Create analysis data for email
            analysis_data: AnalysisData = {
                "config_name": config.name,
                "stories_count": len(story_analyses),
                "from_date": from_date,
                "to_date": to_date,
                "email_receivers_override": args.email_receivers,
            }

            # The execute function expects the regular report path.
            # It will automatically look for the email version in
            # email_reports/
            send_email(Path(regular_report_path), analysis_data)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")
            print("Make sure EMAIL_* environment variables are configured.")
    else:
        print("\nYou can open it in a browser to preview the email template.")
        print("Use --email flag to send the report via email.")


if __name__ == "__main__":
    main()

