"""Tests for RunListView."""

from datetime import date, timedelta

import pytest
from django.test import RequestFactory
from django.utils import timezone
from web.newsserver.models import AnalysisSummary, NewsConfig, ScrapeSummary
from web.newsserver.views import RunListView


@pytest.fixture
def request_factory():
    """Provide a Django RequestFactory."""
    return RequestFactory()


@pytest.mark.django_db
class TestRunListView:
    """Test cases for RunListView."""

    def test_get_context_data_default_date(self, request_factory):
        """Test getting context data with default date (today)."""
        view = RunListView()
        view.request = request_factory.get("/runs/")

        context = view.get_context_data()

        assert "selected_date" in context
        assert context["selected_date"] == timezone.now().date()
        assert "scrape_runs" in context
        assert "analysis_runs" in context

    def test_get_context_data_specific_date(self, request_factory):
        """Test getting context data for a specific date."""
        target_date = date(2025, 1, 1)
        date_str = target_date.isoformat()

        view = RunListView()
        view.request = request_factory.get(f"/runs/?date={date_str}")

        context = view.get_context_data()

        assert context["selected_date"] == target_date

    def test_get_context_data_invalid_date(self, request_factory):
        """Test getting context data with invalid date string."""
        view = RunListView()
        view.request = request_factory.get("/runs/?date=invalid-date")

        context = view.get_context_data()

        # Should fallback to today
        assert context["selected_date"] == timezone.now().date()

    def test_get_context_data_filtering(self, request_factory):
        """Test that runs are filtered by date."""
        config = NewsConfig.objects.create(key="test", display_name="Test")

        # Create run for today
        scrape_today = ScrapeSummary.objects.create(
            config=config, success=True, duration=10
        )

        # Create run for yesterday
        yesterday = timezone.now() - timedelta(days=1)
        scrape_yesterday = ScrapeSummary.objects.create(
            config=config, success=True, duration=10
        )
        scrape_yesterday.timestamp = yesterday
        scrape_yesterday.save()

        # Request for today
        view = RunListView()
        view.request = request_factory.get("/runs/")
        context = view.get_context_data()

        # We need to check IDs because objects might be re-fetched
        scrape_run_ids = [r.id for r in context["scrape_runs"]]
        assert scrape_today.id in scrape_run_ids
        assert scrape_yesterday.id not in scrape_run_ids

        # Request for yesterday
        date_str = yesterday.date().isoformat()
        view.request = request_factory.get(f"/runs/?date={date_str}")
        context = view.get_context_data()

        scrape_run_ids = [r.id for r in context["scrape_runs"]]
        assert scrape_yesterday.id in scrape_run_ids
        assert scrape_today.id not in scrape_run_ids
