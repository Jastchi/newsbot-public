"""Tests for newsserver management commands."""

import pytest
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command

from django.contrib.auth import get_user_model
from web.newsserver.models import Subscriber, SubscriberRequest

User = get_user_model()


@pytest.mark.django_db
class TestCreateTestUserCommand:
    """Tests for create_test_user management command (creates Subscriber, Google-only login)."""

    def test_creates_subscriber_google_only(self):
        """Command creates a Subscriber with given email (password required for login)."""
        out = StringIO()
        call_command(
            "create_test_user",
            email="test-not-subscribed@example.com",
            password="testpass",
            stdout=out,
        )
        out.seek(0)
        text = out.read()
        assert "Created test subscriber" in text or "test-not-subscribed@example.com" in text
        user = User.objects.filter(email="test-not-subscribed@example.com").first()
        assert user is not None
        assert user.is_staff is False
        assert user.is_superuser is False
        assert Subscriber.objects.filter(email__iexact=user.email).count() == 1

    def test_creates_user_custom_email(self):
        """Command accepts --email."""
        out = StringIO()
        call_command(
            "create_test_user",
            email="custom@example.com",
            password="custompass",
            stdout=out,
        )
        user = User.objects.filter(email="custom@example.com").first()
        assert user is not None

    def test_existing_user_no_force_warns(self):
        """When subscriber already exists and --force not set, command warns and does not overwrite."""
        User.objects.create_user(
            email="existing@example.com",
            first_name="",
            last_name="",
            password="ignored",
            is_staff=False,
        )
        out = StringIO()
        call_command(
            "create_test_user",
            email="existing@example.com",
            password="existingpass",
            stdout=out,
        )
        out.seek(0)
        assert "already exists" in out.read().lower()
        user = User.objects.get(email="existing@example.com")
        assert user is not None

    def test_existing_user_force_updates(self):
        """With --force, existing subscriber gets set to non-staff."""
        User.objects.create_user(
            email="force@example.com",
            first_name="",
            last_name="",
            password="ignored",
            is_staff=True,
        )
        out = StringIO()
        call_command(
            "create_test_user",
            email="force@example.com",
            password="forcepass",
            force=True,
            stdout=out,
        )
        user = User.objects.get(email="force@example.com")
        assert user.is_staff is False


@pytest.mark.django_db
class TestSendSubscriberRequestDigestCommand:
    """Tests for send_subscriber_request_digest management command."""

    def test_no_pending_requests(self):
        """When no pending requests, command reports no new requests."""
        out = StringIO()
        call_command("send_subscriber_request_digest", stdout=out)
        out.seek(0)
        assert "No new" in out.read() or "0" in out.read()

    @patch("web.newsserver.management.commands.send_subscriber_request_digest.settings")
    def test_dry_run_lists_requests_without_sending(self, mock_settings):
        """--dry-run lists pending requests without sending or marking."""
        mock_settings.EMAIL_ADMIN_NOTIFICATION_TO = "admin@example.com"
        mock_settings.DEFAULT_FROM_EMAIL = "noreply@example.com"
        SubscriberRequest.objects.create(
            email="dry1@example.com",
            first_name="Dry",
            last_name="One",
        )
        out = StringIO()
        call_command("send_subscriber_request_digest", "--dry-run", stdout=out)
        out.seek(0)
        text = out.read()
        assert "dry1@example.com" in text
        req = SubscriberRequest.objects.get(email="dry1@example.com")
        assert req.included_in_daily_email_at is None

    @patch("web.newsserver.management.commands.send_subscriber_request_digest.settings")
    def test_pending_requests_no_admin_email_warns(self, mock_settings):
        """When pending requests exist but EMAIL_ADMIN_NOTIFICATION_TO is empty, command warns and does not mark."""
        mock_settings.EMAIL_ADMIN_NOTIFICATION_TO = ""
        SubscriberRequest.objects.create(
            email="warn1@example.com",
            first_name="Warn",
            last_name="One",
        )
        out = StringIO()
        call_command("send_subscriber_request_digest", stdout=out)
        out.seek(0)
        text = out.read()
        assert "EMAIL_ADMIN_NOTIFICATION_TO" in text or "not set" in text.lower()
        req = SubscriberRequest.objects.get(email="warn1@example.com")
        assert req.included_in_daily_email_at is None

    def test_send_marks_included_when_admin_set(self):
        """When admin email is set and send_mail is mocked, command sends and marks records."""
        from unittest.mock import MagicMock, patch

        SubscriberRequest.objects.create(
            email="sent1@example.com",
            first_name="Sent",
            last_name="One",
        )
        out = StringIO()
        mock_settings = MagicMock()
        mock_settings.EMAIL_ADMIN_NOTIFICATION_TO = "admin@example.com"
        mock_settings.DEFAULT_FROM_EMAIL = "noreply@example.com"
        with patch(
            "web.newsserver.management.commands.send_subscriber_request_digest.send_mail",
        ) as mock_send:
            with patch(
                "web.newsserver.management.commands.send_subscriber_request_digest.settings",
                mock_settings,
            ):
                call_command("send_subscriber_request_digest", stdout=out)
        mock_send.assert_called_once()
        req = SubscriberRequest.objects.get(email="sent1@example.com")
        assert req.included_in_daily_email_at is not None
        out.seek(0)
        assert "Sent digest" in out.read()
