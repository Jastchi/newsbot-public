"""Tests for newsserver URL configuration."""

import pytest
from django.urls import resolve, reverse
from web.newsserver.views import (
    ConfigOverviewView,
    ConfigReportView,
    LogsView,
    log_stream_view,
)


class TestNewsserverUrls:
    """Test cases for newsserver URL patterns."""

    def test_overview_url_resolves(self):
        """Test that overview URL resolves correctly."""
        url = reverse("newsserver:overview")
        assert url == "/"
        resolved = resolve(url)
        assert resolved.func.view_class == ConfigOverviewView

    def test_config_report_url_resolves(self):
        """Test that config report URL resolves correctly."""
        url = reverse(
            "newsserver:config_report", kwargs={"config_name": "Technology"}
        )
        assert url == "/config/Technology/"
        resolved = resolve(url)
        assert resolved.func.view_class == ConfigReportView

    def test_config_report_url_with_special_chars(self):
        """Test config report URL with special characters."""
        url = reverse(
            "newsserver:config_report", kwargs={"config_name": "World News"}
        )
        resolved = resolve(url)
        assert resolved.func.view_class == ConfigReportView
        # URL encoding converts spaces to %20
        assert resolved.kwargs["config_name"] == "World%20News"

    def test_logs_list_url_resolves(self):
        """Test that logs list URL resolves correctly."""
        url = reverse("newsserver:logs_list")
        assert url == "/logs/"
        resolved = resolve(url)
        assert resolved.func.view_class == LogsView

    def test_logs_stream_url_resolves(self):
        """Test that logs stream URL resolves correctly."""
        url = reverse("newsserver:logs_stream")
        assert url == "/logs/stream/"
        resolved = resolve(url)
        # log_stream_view is a function-based view
        assert callable(resolved.func)
        assert resolved.func == log_stream_view
