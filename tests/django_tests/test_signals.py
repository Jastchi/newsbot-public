"""Tests for newsserver signal handlers."""

from unittest.mock import patch

import pytest
from django.db.models.signals import post_save
from django.test import override_settings

from web.newsserver.models import Subscriber, SubscriberRequest
from web.newsserver.signals import (
    DISPATCH_UID_SUBSCRIBER_ACCEPTED,
    _build_login_url,
    notify_user_when_subscriber_request_accepted,
)


class TestSignalRegistration:
    """The handler must be connected by NewsserverConfig.ready()."""

    def test_handler_connected_on_app_ready(self):
        """A Subscriber post_save receiver is registered at startup."""
        uids = [
            entry[0][0]
            for entry in post_save.receivers
            if isinstance(entry[0], tuple)
        ]
        assert DISPATCH_UID_SUBSCRIBER_ACCEPTED in uids

    def test_handler_connected_only_once(self):
        """dispatch_uid prevents duplicate registration (double emails)."""
        uids = [
            entry[0][0]
            for entry in post_save.receivers
            if isinstance(entry[0], tuple)
        ]
        assert uids.count(DISPATCH_UID_SUBSCRIBER_ACCEPTED) == 1


@pytest.mark.django_db
class TestNotifyUserWhenSubscriberRequestAccepted:
    """Tests for the subscription-accepted notification email."""

    @patch("web.newsserver.signals._send_email")
    def test_email_sent_when_request_accepted(self, mock_send):
        """Creating a Subscriber for a pending request mails the user."""
        SubscriberRequest.objects.create(email="pending@example.com")
        Subscriber.objects.create(
            email="pending@example.com",
            first_name="Pending",
            last_name="User",
        )
        assert mock_send.call_count == 1
        to, subject, message = mock_send.call_args.args[:3]
        assert to == "pending@example.com"
        assert "accepted" in subject.lower()
        assert "log in" in message.lower()

    @patch("web.newsserver.signals._send_email")
    def test_admin_accept_action_sends_email(self, mock_send):
        """The admin 'Accept selected' action triggers the email.

        The action deletes the request right after creating the
        Subscriber, so this also pins the ordering the handler relies on.
        """
        from django.contrib.admin.sites import AdminSite

        from web.newsserver.admin import SubscriberRequestAdmin

        SubscriberRequest.objects.create(email="approved@example.com")
        model_admin = SubscriberRequestAdmin(SubscriberRequest, AdminSite())
        with patch.object(model_admin, "message_user"):
            model_admin.accept_requests_create_subscriber(
                None,
                SubscriberRequest.objects.filter(email="approved@example.com"),
            )
        assert Subscriber.objects.filter(
            email="approved@example.com",
        ).exists()
        assert mock_send.call_args.args[0] == "approved@example.com"

    @patch("web.newsserver.signals._send_email")
    def test_no_email_without_pending_request(self, mock_send):
        """A Subscriber created out of the blue triggers no email."""
        Subscriber.objects.create(
            email="direct@example.com",
            first_name="Direct",
            last_name="User",
        )
        mock_send.assert_not_called()

    @patch("web.newsserver.signals._send_email")
    def test_no_email_on_update(self, mock_send):
        """Only creation notifies; later saves must stay silent."""
        SubscriberRequest.objects.create(email="pending@example.com")
        sub = Subscriber.objects.create(
            email="pending@example.com",
            first_name="Pending",
            last_name="User",
        )
        mock_send.reset_mock()
        sub.first_name = "Renamed"
        sub.save()
        mock_send.assert_not_called()

    @patch("web.newsserver.signals._send_email", side_effect=OSError("smtp"))
    def test_send_failure_does_not_break_creation(self, _mock_send):
        """A failing mail server must not roll back the Subscriber."""
        SubscriberRequest.objects.create(email="pending@example.com")
        Subscriber.objects.create(
            email="pending@example.com",
            first_name="Pending",
            last_name="User",
        )
        assert Subscriber.objects.filter(
            email="pending@example.com",
        ).exists()


class TestBuildLoginUrl:
    """The login link must target the host that serves the app."""

    @override_settings(
        APP_BASE_URL="https://app.thenewsbot.net",
        CANONICAL_SITE_URL="https://thenewsbot.net",
    )
    def test_prefers_app_base_url(self):
        """In a subdomain split the link uses the app host."""
        assert _build_login_url().startswith("https://app.thenewsbot.net/")

    @override_settings(
        APP_BASE_URL="",
        CANONICAL_SITE_URL="https://thenewsbot.net",
    )
    def test_falls_back_to_canonical_site(self):
        """Single-host deploys use the canonical site URL."""
        assert _build_login_url().startswith("https://thenewsbot.net/")

    @override_settings(APP_BASE_URL="", CANONICAL_SITE_URL="")
    @pytest.mark.django_db
    def test_falls_back_to_sites_domain(self):
        """With nothing configured, the sites framework domain is used."""
        url = _build_login_url()
        assert url.startswith(("http://", "https://"))
        assert url.endswith("/accounts/login/")


@pytest.mark.django_db
def test_handler_ignores_blank_email():
    """A Subscriber without an email never triggers a lookup or mail."""
    with patch("web.newsserver.signals._send_email") as mock_send:
        notify_user_when_subscriber_request_accepted(
            instance=Subscriber(email=""),
            created=True,
        )
    mock_send.assert_not_called()
