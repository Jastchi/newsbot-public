"""Tests for the test_data module."""

import pytest

from newsbot.test_data import TEST_ARTICLES, insert_test_articles

from utilities.django_models import Article as DjangoArticle


@pytest.fixture(autouse=True)
def cleanup_articles(db):
    """Cleanup test articles after each test."""
    yield
    DjangoArticle.objects.all().delete()


@pytest.mark.django_db
def test_insert_test_articles_clears_existing():
    """Test that insert_test_articles clears existing articles for same config."""
    config_key = "test_config"

    # First, insert some articles manually for the same config
    DjangoArticle.objects.create(
        config_file=config_key,
        title="Existing Article",
        content="Content",
        source="Test",
        url="https://existing.com/1",
    )

    # Verify article exists
    count = DjangoArticle.objects.count()
    assert count == 1

    # Now insert test articles for the same config
    inserted = insert_test_articles(config_key)

    # Should have exactly the test articles, not the existing one
    count = DjangoArticle.objects.count()
    assert count == len(TEST_ARTICLES)
    assert count == inserted


@pytest.mark.django_db
def test_insert_test_articles_returns_count():
    """Test that insert_test_articles returns the correct count."""
    count = insert_test_articles()
    assert count == len(TEST_ARTICLES)
    assert count == 5


@pytest.mark.django_db
def test_insert_test_articles_creates_correct_articles():
    """Test that inserted articles have correct data."""
    config_key = "test_config"
    insert_test_articles(config_key)

    articles = DjangoArticle.objects.all()

    # Check we have articles from expected sources
    sources = {a.source for a in articles}
    assert "Example Source 1" in sources
    assert "Example Source 2" in sources
    assert "Example Source 3" in sources

    # Check all articles have required fields
    for article in articles:
        assert article.config_file == config_key
        assert article.title
        assert article.content
        assert article.source
        assert article.url
        assert article.published_date
        assert article.scraped_date


def test_test_articles_constant_has_expected_structure():
    """Test that TEST_ARTICLES has the expected structure."""
    assert len(TEST_ARTICLES) == 5

    for article in TEST_ARTICLES:
        assert "title" in article
        assert "content" in article
        assert "source" in article
        assert "url" in article

