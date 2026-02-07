import json
from datetime import UTC, datetime
from django.urls import reverse
from django.utils import timezone
import pytest

from newsbot.constants import TZ
from web.newsserver.models import NewsConfig

@pytest.mark.django_db
class TestNewsSchedulerDashboardView:
    @pytest.fixture(autouse=True)
    def setup(self, client, admin_user, sample_news_configs):
        self.client = client
        self.admin_user = admin_user
        self.url = reverse("newsserver:news_schedule")
        self.configs = sample_news_configs

    def test_access_control_authenticated_can_view(self):
        """Test that any authenticated user can view the dashboard; staff can also edit."""
        from django.contrib.auth import get_user_model
        User = get_user_model()

        # Non-staff user can view the schedule (read-only)
        user = User.objects.create_user(
            email="testuser@example.com",
            password="password",
        )
        self.client.force_login(user)
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert "newsserver/news_scheduler_calendar.html" in [t.name for t in response.templates]
        # Template receives user_can_edit_schedule=False for non-staff
        assert response.context["user_can_edit_schedule"] is False

        # Staff user can view and edit the schedule
        self.admin_user.is_staff = True
        self.admin_user.save()
        self.client.force_login(self.admin_user)
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert response.context["user_can_edit_schedule"] is True

    def test_get_standard_request(self):
        """Test that a standard GET request returns the calendar page."""
        self.admin_user.is_staff = True
        self.admin_user.save()
        self.client.force_login(self.admin_user)
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert "newsserver/news_scheduler_calendar.html" in [t.name for t in response.templates]

    def test_get_ajax_data(self):
        """Test that an AJAX GET request returns JSON event data."""
        self.admin_user.is_staff = True
        self.admin_user.save()
        self.client.force_login(self.admin_user)
        
        # Enable scheduler for one config
        config = self.configs[0]
        config.scheduler_weekly_analysis_enabled = True
        config.scheduler_weekly_analysis_day_of_week = "mon"
        config.scheduler_weekly_analysis_hour = 10
        config.scheduler_weekly_analysis_minute = 0
        config.save()

        # Mock start and end params as FullCalendar would send them
        now = timezone.now()
        start = (now - timezone.timedelta(days=7)).isoformat()
        end = (now + timezone.timedelta(days=7)).isoformat()

        response = self.client.get(self.url, {"ajax": "1", "start": start, "end": end})
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Verify event structure
        event = data[0]
        assert "id" in event
        assert "title" in event
        assert "start" in event

    def test_post_update_config(self):
        """Test that a POST request updates the NewsConfig scheduler settings."""
        self.admin_user.is_staff = True
        self.admin_user.save()
        self.client.force_login(self.admin_user)
        config = self.configs[0]

        # Frontend sends UTC (toISOString()). Use same conversion as view.
        new_start = "2026-01-27T13:30:00Z"
        utc_dt = datetime(2026, 1, 27, 13, 30, 0, tzinfo=UTC)
        local_dt = utc_dt.astimezone(TZ)
        day_map = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}

        update_data = {"id": config.id, "start": new_start}

        response = self.client.post(
            self.url,
            data=json.dumps(update_data),
            content_type="application/json",
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

        config.refresh_from_db()
        assert config.scheduler_weekly_analysis_day_of_week == day_map[local_dt.weekday()]
        assert config.scheduler_weekly_analysis_hour == local_dt.hour
        assert config.scheduler_weekly_analysis_minute == local_dt.minute

    def test_post_invalid_data(self):
        """Test that invalid POST data returns an error."""
        self.admin_user.is_staff = True
        self.admin_user.save()
        self.client.force_login(self.admin_user)
        
        # Missing ID
        response = self.client.post(
            self.url,
            data=json.dumps({"start": "2026-01-27T14:30:00Z"}),
            content_type="application/json"
        )
        assert response.status_code == 400
        
        # Invalid config ID
        response = self.client.post(
            self.url,
            data=json.dumps({"id": 99999, "start": "2026-01-27T14:30:00Z"}),
            content_type="application/json"
        )
        assert response.status_code == 400
