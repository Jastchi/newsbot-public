"""Tests for newsserver admin."""

from unittest.mock import Mock

import pytest
from django.contrib.admin.sites import AdminSite
from django.test import RequestFactory
from web.newsserver.admin import (
    NewsConfigAdmin,
    SubscriberAdmin,
    SubscriberRequestAdmin,
)
from web.newsserver.models import NewsConfig, Subscriber, SubscriberRequest


@pytest.fixture
def request_factory():
    """Provide a Django RequestFactory."""
    return RequestFactory()


@pytest.fixture
def admin_site():
    """Provide an AdminSite instance."""
    return AdminSite()


@pytest.mark.django_db
class TestNewsConfigAdmin:
    """Test cases for NewsConfigAdmin."""

    def test_list_display(self, admin_site):
        """Test that list_display is configured correctly."""
        admin = NewsConfigAdmin(NewsConfig, admin_site)
        assert "display_name" in admin.list_display
        assert "key" in admin.list_display
        assert "subscriber_count" in admin.list_display
        assert "created_at_date" in admin.list_display

    def test_created_at_date_display(self, admin_site):
        """Test created_at_date method."""
        config = NewsConfig.objects.create(
            key="test",
            display_name="Test Config",
        )
        admin = NewsConfigAdmin(NewsConfig, admin_site)
        result = admin.created_at_date(config)
        assert "-" in result  # Should be formatted as YYYY-MM-DD

    def test_subscriber_count_display_zero(self, admin_site):
        """Test subscriber_count with no subscribers."""
        config = NewsConfig.objects.create(
            key="test",
            display_name="Test Config",
        )
        admin = NewsConfigAdmin(NewsConfig, admin_site)
        result = admin.subscriber_count(config)
        assert result == "0 subscribers"

    def test_subscriber_count_display_one(self, admin_site):
        """Test subscriber_count with one subscriber."""
        config = NewsConfig.objects.create(
            key="test",
            display_name="Test Config",
        )
        subscriber = Subscriber.objects.create_user(
            email="john@example.com",
            first_name="John",
            last_name="Doe",
            password="testpass",
        )
        subscriber.configs.add(config)

        admin = NewsConfigAdmin(NewsConfig, admin_site)
        result = admin.subscriber_count(config)
        assert result == "1 subscriber"

    def test_subscriber_count_display_multiple(self, admin_site):
        """Test subscriber_count with multiple subscribers."""
        config = NewsConfig.objects.create(
            key="test",
            display_name="Test Config",
        )
        for i in range(3):
            subscriber = Subscriber.objects.create_user(
                email=f"user{i}@example.com",
                first_name=f"User{i}",
                last_name="Test",
                password="testpass",
            )
            subscriber.configs.add(config)

        admin = NewsConfigAdmin(NewsConfig, admin_site)
        result = admin.subscriber_count(config)
        assert result == "3 subscribers"

    def test_search_fields(self, admin_site):
        """Test that search_fields includes key and display_name."""
        admin = NewsConfigAdmin(NewsConfig, admin_site)
        assert "key" in admin.search_fields
        assert "display_name" in admin.search_fields


@pytest.mark.django_db
class TestSubscriberAdmin:
    """Test cases for SubscriberAdmin."""

    def test_list_display(self, admin_site):
        """Test that list_display is configured correctly."""
        admin = SubscriberAdmin(Subscriber, admin_site)
        assert "full_name" in admin.list_display
        assert "email" in admin.list_display
        assert "is_active" in admin.list_display
        assert "config_count" in admin.list_display

    def test_full_name_display(self, admin_site):
        """Test full_name method."""
        subscriber = Subscriber.objects.create(
            first_name="Jane",
            last_name="Smith",
            email="jane@example.com",
        )
        admin = SubscriberAdmin(Subscriber, admin_site)
        result = admin.full_name(subscriber)
        assert result == "Jane Smith"

    def test_config_count_display_zero(self, admin_site):
        """Test config_count with no configs."""
        subscriber = Subscriber.objects.create(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
        )
        admin = SubscriberAdmin(Subscriber, admin_site)
        result = admin.config_count(subscriber)
        assert result == "0 configs"

    def test_config_count_display_one(self, admin_site):
        """Test config_count with one config."""
        subscriber = Subscriber.objects.create(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
        )
        config = NewsConfig.objects.create(
            key="test",
            display_name="Test Config",
        )
        subscriber.configs.add(config)

        admin = SubscriberAdmin(Subscriber, admin_site)
        result = admin.config_count(subscriber)
        assert result == "1 config"

    def test_config_count_display_multiple(self, admin_site):
        """Test config_count with multiple configs."""
        subscriber = Subscriber.objects.create(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
        )
        for i in range(3):
            config = NewsConfig.objects.create(
                key=f"test{i}",
                display_name=f"Test Config {i}",
            )
            subscriber.configs.add(config)

        admin = SubscriberAdmin(Subscriber, admin_site)
        result = admin.config_count(subscriber)
        assert result == "3 configs"

    def test_list_editable(self, admin_site):
        """Test that is_active is editable in list view."""
        admin = SubscriberAdmin(Subscriber, admin_site)
        assert "is_active" in admin.list_editable

    def test_filter_horizontal(self, admin_site):
        """Test that configs uses filter_horizontal widget."""
        admin = SubscriberAdmin(Subscriber, admin_site)
        assert "configs" in admin.filter_horizontal


@pytest.mark.django_db
class TestSubscriberRequestAdmin:
    """Test cases for SubscriberRequestAdmin."""

    def test_list_display(self, admin_site):
        """Test that list_display is configured correctly."""
        admin = SubscriberRequestAdmin(SubscriberRequest, admin_site)
        assert "email" in admin.list_display
        assert "first_name" in admin.list_display
        assert "last_name" in admin.list_display
        assert "created_at" in admin.list_display
        assert "included_in_daily_email_at" in admin.list_display

    def test_included_display_not_sent(self, admin_site):
        """Test included method when not yet in digest."""
        req = SubscriberRequest.objects.create(
            email="req@example.com",
            first_name="Req",
            last_name="User",
        )
        admin = SubscriberRequestAdmin(SubscriberRequest, admin_site)
        result = admin.included(req)
        assert result == "No"

    def test_included_display_sent(self, admin_site):
        """Test included method when already in digest."""
        from django.utils import timezone
        req = SubscriberRequest.objects.create(
            email="sent@example.com",
            first_name="Sent",
            last_name="User",
            included_in_daily_email_at=timezone.now(),
        )
        admin = SubscriberRequestAdmin(SubscriberRequest, admin_site)
        result = admin.included(req)
        assert result == "Yes"
