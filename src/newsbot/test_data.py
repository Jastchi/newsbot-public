"""
Test data for pipeline testing.

Provides deterministic test articles for analysis testing.
5 articles that cluster into 2 stories:
- Cluster 1 (3 articles): AI development negotiations
- Cluster 2 (2 articles): Technology industry
"""

import logging
from datetime import datetime, timedelta

from newsbot.constants import TZ
from utilities import setup_django
from utilities.django_models import Article as DjangoArticle
from utilities.django_models import NewsConfig

logger = logging.getLogger(__name__)

# Test articles designed to cluster into 2 groups based on titles
TEST_ARTICLES = [
    # Cluster 1: AI development (3 articles from 3 sources)
    {
        "title": "AI Safety Talks Resume as Regulators Push for Framework",
        "content": (
            "Negotiations for AI safety regulations between major tech "
            "companies and government agencies have resumed with "
            "international bodies working to bridge gaps between industry "
            "and regulatory perspectives."
        ),
        "source": "Example Source 1",
        "url": "https://test.example.com/article-1",
    },
    {
        "title": "AI Regulation Negotiations Enter Critical Phase",
        "content": (
            "International regulators report progress in AI governance "
            "discussions as both tech companies and governments consider "
            "new proposals for establishing safety standards."
        ),
        "source": "Example Source 2",
        "url": "https://test.example.com/article-2",
    },
    {
        "title": "AI Safety Framework Could Be Reached Soon, Officials Say",
        "content": (
            "Senior officials involved in the negotiations express optimism "
            "that an AI safety agreement may be finalized in the coming days."
        ),
        "source": "Example Source 3",
        "url": "https://test.example.com/article-3",
    },
    # Cluster 2: Tech industry (2 articles from 2 sources)
    {
        "title": "Tech Startup Raises $50 Million in Funding",
        "content": (
            "A Silicon Valley-based AI startup has secured significant "
            "investment from international venture capital firms."
        ),
        "source": "Example Source 1",
        "url": "https://test.example.com/article-4",
    },
    {
        "title": "Tech Startup Announces Major Investment",
        "content": (
            "The technology sector continues to attract global "
            "investors with another startup securing substantial funding."
        ),
        "source": "Example Source 2",
        "url": "https://test.example.com/article-5",
    },
]


def insert_test_articles(config_key: str = "test") -> int:
    """
    Clear all articles for the config and insert test articles.

    Requires Django to be set up before calling.

    Args:
        config_key: Config key (e.g., "test") to associate
            with test articles.

    Returns:
        Number of articles inserted

    """
    setup_django()

    # Look up NewsConfig by key
    news_config = NewsConfig.objects.filter(key=config_key).first()

    # Clear existing articles for this config
    # (using both FK and legacy field)
    if news_config is not None:
        deleted, _ = DjangoArticle.objects.filter(
            config=news_config,
        ).delete()
    else:
        deleted, _ = DjangoArticle.objects.filter(
            config_file=config_key,
        ).delete()
    logger.info(
        "Deleted %d existing articles for config %s", deleted, config_key,
    )

    # Insert test articles
    now = datetime.now(TZ)
    for i, article_data in enumerate(TEST_ARTICLES):
        DjangoArticle.objects.create(
            config=news_config,  # Use FK (may be None)
            config_file=config_key,  # Keep for backward compatibility
            title=article_data["title"],
            content=article_data["content"],
            source=article_data["source"],
            url=article_data["url"],
            published_date=now - timedelta(hours=i),
            scraped_date=now,
        )

    logger.info(
        f"Inserted {len(TEST_ARTICLES)} test articles for config "
        f"{config_key}",
    )

    return len(TEST_ARTICLES)

