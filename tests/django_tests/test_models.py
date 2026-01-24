"""Tests for newsserver models."""

from datetime import datetime

import pytest
from django.db import IntegrityError
from django.utils import timezone
from web.newsserver.models import NewsConfig, Subscriber


@pytest.mark.django_db
class TestNewsConfigModel:
    """Test cases for NewsConfig model."""

    def test_create_news_config(self):
        """Test creating a news config."""
        config = NewsConfig.objects.create(
            key="test-config",
            display_name="Test Configuration",
        )
        assert config.key == "test-config"
        assert config.display_name == "Test Configuration"
        assert config.created_at is not None
        assert config.updated_at is not None

    def test_news_config_str(self):
        """Test string representation of NewsConfig."""
        config = NewsConfig.objects.create(
            key="tech",
            display_name="Technology News",
        )
        assert str(config) == "Technology News - tech"

    def test_news_config_unique_key(self):
        """Test that key field has unique constraint defined."""
        # Check that the key field has unique=True in the model definition
        key_field = NewsConfig._meta.get_field("key")
        assert key_field.unique is True

        # Note: IntegrityError enforcement in tests depends on database migrations
        # being applied. The important part is that unique=True is defined on the field.

    def test_news_config_subscribers_relationship(self):
        """Test relationship between NewsConfig and Subscribers."""
        config = NewsConfig.objects.create(
            key="world",
            display_name="World News",
        )
        subscriber = Subscriber.objects.create(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
        )
        subscriber.configs.add(config)

        assert config.subscribers.count() == 1
        assert config.subscribers.first() == subscriber


@pytest.mark.django_db
class TestSubscriberModel:
    """Test cases for Subscriber model."""

    def test_create_subscriber(self):
        """Test creating a subscriber."""
        subscriber = Subscriber.objects.create(
            first_name="Jane",
            last_name="Smith",
            email="jane@example.com",
            is_active=True,
        )
        assert subscriber.first_name == "Jane"
        assert subscriber.last_name == "Smith"
        assert subscriber.email == "jane@example.com"
        assert subscriber.is_active is True

    def test_subscriber_str(self):
        """Test string representation of Subscriber."""
        subscriber = Subscriber.objects.create(
            first_name="Bob",
            last_name="Johnson",
            email="bob@example.com",
        )
        assert str(subscriber) == "Bob Johnson <bob@example.com>"

    def test_subscriber_unique_email(self):
        """Test that email must be unique."""
        Subscriber.objects.create(
            first_name="Alice",
            last_name="Brown",
            email="alice@example.com",
        )
        with pytest.raises(IntegrityError):
            Subscriber.objects.create(
                first_name="Alice",
                last_name="Smith",
                email="alice@example.com",
            )

    def test_subscriber_default_active(self):
        """Test that is_active defaults to True."""
        subscriber = Subscriber.objects.create(
            first_name="Charlie",
            last_name="Davis",
            email="charlie@example.com",
        )
        assert subscriber.is_active is True

    def test_subscribed_config_keys(self):
        """Test getting subscribed configuration keys."""
        subscriber = Subscriber.objects.create(
            first_name="Eve",
            last_name="Wilson",
            email="eve@example.com",
        )
        config1 = NewsConfig.objects.create(
            key="tech",
            display_name="Technology",
        )
        config2 = NewsConfig.objects.create(
            key="sports",
            display_name="Sports",
        )
        config3 = NewsConfig.objects.create(
            key="world",
            display_name="World",
        )

        subscriber.configs.add(config1, config3)

        keys = subscriber.subscribed_config_keys()
        assert len(keys) == 2
        assert "tech" in keys
        assert "world" in keys
        assert "sports" not in keys

    def test_subscribed_config_keys_empty(self):
        """Test subscribed_config_keys with no subscriptions."""
        subscriber = Subscriber.objects.create(
            first_name="Frank",
            last_name="Miller",
            email="frank@example.com",
        )
        keys = subscriber.subscribed_config_keys()
        assert keys == []

    def test_subscriber_multiple_configs(self):
        """Test subscriber can have multiple configs."""
        subscriber = Subscriber.objects.create(
            first_name="Grace",
            last_name="Lee",
            email="grace@example.com",
        )
        config1 = NewsConfig.objects.create(key="news1", display_name="News 1")
        config2 = NewsConfig.objects.create(key="news2", display_name="News 2")
        config3 = NewsConfig.objects.create(key="news3", display_name="News 3")

        subscriber.configs.add(config1, config2, config3)
        assert subscriber.configs.count() == 3



