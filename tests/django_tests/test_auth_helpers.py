"""Tests for web.newsserver.auth_helpers (magic-link auth and admin notification)."""

from unittest.mock import patch

import pytest
from django.conf import settings
from django.core.cache import cache
from django.test import RequestFactory, override_settings
from django.urls import reverse

from web.newsserver.auth_helpers import (
    MAGIC_LINK_RATE_LIMIT_MAX,
    MAGIC_LINK_TOKEN_TIMEOUT_MINUTES,
    build_magic_link_verify_url,
    consume_magic_link_token,
    create_magic_link_token,
    get_client_ip,
    magic_link_rate_limit_exceeded,
    magic_link_sent,
    magic_link_signup_requested,
    notify_admin_subscriber_request,
    request_magic_link,
    send_magic_link_email,
    verify_magic_link,
)
from web.newsserver.models import Subscriber, SubscriberRequest


# ----- get_client_ip -----


class TestGetClientIp:
    """Tests for get_client_ip."""

    def test_uses_remote_addr_when_no_forwarded_for(self, request_factory):
        request = request_factory.get("/")
        request.META = {"REMOTE_ADDR": "192.168.1.1"}
        assert get_client_ip(request) == "192.168.1.1"

    def test_uses_first_x_forwarded_for_when_present(self, request_factory):
        request = request_factory.get("/")
        request.META = {
            "REMOTE_ADDR": "10.0.0.1",
            "HTTP_X_FORWARDED_FOR": "203.0.113.50, 70.41.3.18",
        }
        assert get_client_ip(request) == "203.0.113.50"

    def test_strips_whitespace_from_x_forwarded_for(self, request_factory):
        request = request_factory.get("/")
        request.META = {"HTTP_X_FORWARDED_FOR": "  203.0.113.50  "}
        assert get_client_ip(request) == "203.0.113.50"


# ----- magic_link_rate_limit_exceeded -----


@pytest.fixture
def request_factory():
    return RequestFactory()


class TestMagicLinkRateLimitExceeded:
    """Tests for magic_link_rate_limit_exceeded."""

    def setup_method(self):
        """Clear cache so rate limit counts are predictable."""
        try:
            cache.clear()
        except Exception:
            pass

    def test_empty_email_returns_true(self, request_factory):
        request = request_factory.get("/")
        request.META = {"REMOTE_ADDR": "1.2.3.4"}
        assert magic_link_rate_limit_exceeded(request, "") is True
        assert magic_link_rate_limit_exceeded(request, "   ") is True

    def test_under_limit_returns_false_and_consumes_slot(self, request_factory):
        request = request_factory.get("/")
        request.META = {"REMOTE_ADDR": "5.6.7.8"}
        email = "first@example.com"
        assert magic_link_rate_limit_exceeded(request, email) is False
        # Same email again within limit count
        assert magic_link_rate_limit_exceeded(request, email) is False
        # Third time exceeds (MAGIC_LINK_RATE_LIMIT_MAX is 2)
        assert magic_link_rate_limit_exceeded(request, email) is True

    def test_different_emails_same_ip_share_ip_limit(self, request_factory):
        # Limit is 2 per IP and 2 per email; first two requests (any emails) use up IP.
        request = request_factory.get("/")
        request.META = {"REMOTE_ADDR": "10.10.10.10"}
        assert magic_link_rate_limit_exceeded(request, "a@example.com") is False
        assert magic_link_rate_limit_exceeded(request, "b@example.com") is False
        # Third request from same IP is rate limited (IP limit shared across emails)
        assert magic_link_rate_limit_exceeded(request, "a@example.com") is True
        assert magic_link_rate_limit_exceeded(request, "b@example.com") is True

    def test_email_normalized_to_lower(self, request_factory):
        request = request_factory.get("/")
        request.META = {"REMOTE_ADDR": "11.11.11.11"}
        assert magic_link_rate_limit_exceeded(request, "User@Example.COM") is False
        assert magic_link_rate_limit_exceeded(request, "user@example.com") is False
        assert magic_link_rate_limit_exceeded(request, "user@example.com") is True


# ----- create_magic_link_token / consume_magic_link_token -----


class TestMagicLinkToken:
    """Tests for create_magic_link_token and consume_magic_link_token."""

    def test_create_returns_token_and_consume_returns_email(self):
        email = "tokenuser@example.com"
        token = create_magic_link_token(email)
        assert isinstance(token, str)
        assert len(token) > 0
        assert consume_magic_link_token(token) == email.strip().lower()

    def test_consume_deletes_token(self):
        email = "once@example.com"
        token = create_magic_link_token(email)
        assert consume_magic_link_token(token) == "once@example.com"
        assert consume_magic_link_token(token) is None

    def test_consume_invalid_token_returns_none(self):
        assert consume_magic_link_token("invalid-token-xyz") is None

    def test_create_normalizes_email_to_lower(self):
        token = create_magic_link_token("  Upper@Example.COM  ")
        assert consume_magic_link_token(token) == "upper@example.com"


# ----- build_magic_link_verify_url -----


class TestBuildMagicLinkVerifyUrl:
    """Tests for build_magic_link_verify_url."""

    @override_settings(SITE_DOMAIN="", CANONICAL_SITE_URL="")
    def test_builds_absolute_url_when_host_set(self, request_factory):
        request = request_factory.get("/")
        request.get_host = lambda: "example.com"
        url = build_magic_link_verify_url(request, "abc123", next_url="")
        assert "abc123" in url
        assert url.startswith("http")
        assert "example.com" in url

    def test_appends_next_param_when_provided(self, request_factory):
        request = request_factory.get("/")
        request.get_host = lambda: "testserver"
        url = build_magic_link_verify_url(
            request, "tok", next_url="/report-archive/"
        )
        assert "next=" in url
        assert "/report-archive/" in url or "report-archive" in url

    def test_relative_path_when_no_host(self, request_factory):
        request = request_factory.get("/")
        request.get_host = lambda: ""
        url = build_magic_link_verify_url(request, "xyz", next_url="")
        assert "xyz" in url
        assert reverse("magic_link_verify", kwargs={"token": "xyz"}) in url or "xyz" in url

    @override_settings(
        SITE_DOMAIN="thenewsbot.net",
        CANONICAL_SITE_URL="https://thenewsbot.net",
        FORCE_SCRIPT_NAME="",
    )
    def test_uses_canonical_site_url_when_configured(self, request_factory):
        request = request_factory.get("/")
        request.get_host = lambda: "www.thenewsbot.net"
        url = build_magic_link_verify_url(request, "abc123", next_url="")
        assert url == (
            "https://thenewsbot.net/accounts/login-by-email/verify/abc123/"
        )


# ----- send_magic_link_email -----


class TestSendMagicLinkEmail:
    """Tests for send_magic_link_email."""

    @patch("web.newsserver.auth_helpers._send_email")
    def test_sends_email_with_subject_and_recipient(self, mock_send):
        send_magic_link_email("recipient@example.com", "https://example.com/verify/abc/")
        mock_send.assert_called_once()
        to, subject, text, html = mock_send.call_args.args
        assert "login" in subject.lower() or "link" in subject.lower()
        assert to == "recipient@example.com"
        assert "https://example.com/verify/abc/" in text
        assert str(MAGIC_LINK_TOKEN_TIMEOUT_MINUTES) in text
        assert html  # template rendered


# ----- notify_admin_subscriber_request -----


@pytest.mark.django_db
class TestNotifyAdminSubscriberRequest:
    """Tests for notify_admin_subscriber_request."""

    @patch("web.newsserver.auth_helpers._send_email")
    def test_sends_email_when_admin_configured(self, mock_send):
        with patch.object(
            settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@newsbot.com"
        ):
            req = SubscriberRequest.objects.create(
                email="newuser@example.com",
                first_name="New",
                last_name="User",
            )
            notify_admin_subscriber_request(req)
        mock_send.assert_called_once()
        to, subject, text = mock_send.call_args.args[:3]
        assert to == "admin@newsbot.com"
        assert "New subscription request" in subject
        assert "newuser@example.com" in text
        assert "New User" in text
        req.refresh_from_db()
        assert req.admin_notified_at is not None

    @patch("web.newsserver.auth_helpers._send_email")
    def test_skips_when_admin_notified_at_already_set(self, mock_send):
        from django.utils import timezone

        with patch.object(
            settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@newsbot.com"
        ):
            req = SubscriberRequest.objects.create(
                email="again@example.com",
                first_name="A",
                last_name="B",
                admin_notified_at=timezone.now(),
            )
            notify_admin_subscriber_request(req)
        mock_send.assert_not_called()

    @patch("web.newsserver.auth_helpers._send_email")
    def test_skips_when_email_admin_unset(self, mock_send):
        with patch.object(settings, "EMAIL_ADMIN_NOTIFICATION_TO", ""):
            req = SubscriberRequest.objects.create(
                email="noadmin@example.com",
                first_name="No",
                last_name="Admin",
            )
            notify_admin_subscriber_request(req)
        mock_send.assert_not_called()

    @patch("web.newsserver.auth_helpers._send_email")
    def test_name_fallback_when_empty(self, mock_send):
        with patch.object(
            settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@newsbot.com"
        ):
            req = SubscriberRequest.objects.create(
                email="noname@example.com",
                first_name="",
                last_name="",
            )
            notify_admin_subscriber_request(req)
        _, _, text = mock_send.call_args.args[:3]
        assert "(no name)" in text


# ----- request_magic_link (view) -----


@pytest.mark.django_db
class TestRequestMagicLinkView:
    """Tests for request_magic_link view."""

    def test_get_returns_form(self, client):
        response = client.get(reverse("magic_link_request"))
        assert response.status_code == 200
        assert "magic_link_valid_minutes" in response.context or b"login" in response.content.lower()

    def test_get_passes_next_from_query(self, client):
        response = client.get(
            reverse("magic_link_request"), {"next": "/report-archive/"}
        )
        assert response.status_code == 200
        assert response.context.get("next") == "/report-archive/" or b"next" in response.content

    def test_post_empty_email_returns_400(self, client):
        response = client.post(
            reverse("magic_link_request"),
            data={"email": ""},
        )
        assert response.status_code == 400
        assert "error" in response.context or b"enter" in response.content.lower()

    def test_post_whitespace_only_email_returns_400(self, client):
        response = client.post(
            reverse("magic_link_request"),
            data={"email": "   "},
        )
        assert response.status_code == 400

    @patch("web.newsserver.auth_helpers._send_email")
    def test_post_valid_email_redirects_and_sends_email(self, mock_send, client):
        response = client.post(
            reverse("magic_link_request"),
            data={"email": "valid@example.com"},
        )
        assert response.status_code == 302
        assert "magic_link_sent" in response.url or "login-by-email/sent" in response.url
        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == "valid@example.com"

    @patch("web.newsserver.auth_helpers._send_email")
    def test_post_valid_email_with_next_redirects_with_next_param(self, mock_send, client):
        response = client.post(
            reverse("magic_link_request"),
            data={"email": "nextuser@example.com", "next": "/report-archive/"},
        )
        assert response.status_code == 302
        assert "next=" in response.url

    @patch("web.newsserver.auth_helpers._send_email")
    def test_post_rate_limited_returns_429(self, mock_send, client):
        cache.clear()
        url = reverse("magic_link_request")
        # Exhaust rate limit for this IP/email (MAGIC_LINK_RATE_LIMIT_MAX is 2)
        for _ in range(MAGIC_LINK_RATE_LIMIT_MAX):
            client.post(url, data={"email": "ratelimit@example.com"})
        response = client.post(url, data={"email": "ratelimit@example.com"})
        assert response.status_code == 429
        assert "error" in response.context or b"Too many" in response.content or b"minutes" in response.content


# ----- magic_link_sent -----


@pytest.mark.django_db
class TestMagicLinkSentView:
    """Tests for magic_link_sent view."""

    def test_returns_200_and_template(self, client):
        response = client.get(reverse("magic_link_sent"))
        assert response.status_code == 200
        template_names = [t.name for t in response.templates]
        assert "account/magic_link_sent.html" in template_names

    def test_passes_next_from_query(self, client):
        response = client.get(
            reverse("magic_link_sent"), {"next": "/somewhere/"}
        )
        assert response.status_code == 200
        assert response.context.get("next") == "/somewhere/"


# ----- verify_magic_link -----


@pytest.mark.django_db
class TestVerifyMagicLinkView:
    """Tests for verify_magic_link view."""

    def test_invalid_token_redirects_to_login_with_error(self, client):
        response = client.get(
            reverse("magic_link_verify", kwargs={"token": "invalid-token-xyz"}),
            follow=False,
        )
        assert response.status_code == 302
        assert "login" in response.url or "accounts" in response.url
        assert "magic_link_invalid" in response.url or "error=" in response.url

    @patch("web.newsserver.auth_helpers.send_mail")
    def test_valid_token_existing_subscriber_logs_in_and_redirects(
        self, mock_send_mail, client
    ):
        from web.newsserver.models import Subscriber

        Subscriber.objects.create_user(
            email="existing@example.com",
            password="unused",
            first_name="Existing",
            last_name="User",
            is_staff=False,
        )
        token = create_magic_link_token("existing@example.com")
        response = client.get(
            reverse("magic_link_verify", kwargs={"token": token}),
            follow=False,
        )
        assert response.status_code == 302
        assert "magic_link_signup" not in response.url
        # User should be logged in; redirect to LOGIN_REDIRECT or next
        req = SubscriberRequest.objects.filter(email__iexact="existing@example.com")
        assert not req.exists()  # no SubscriberRequest created for existing sub

    @patch("web.newsserver.auth_helpers._send_email")
    def test_valid_token_no_subscriber_creates_request_and_redirects_to_signup_requested(
        self, mock_send, client
    ):
        with patch.object(
            settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@newsbot.com"
        ):
            token = create_magic_link_token("newuser@example.com")
            response = client.get(
                reverse("magic_link_verify", kwargs={"token": token}),
                follow=False,
            )
        assert response.status_code == 302
        assert "signup-requested" in response.url or "signup_requested" in response.url
        req = SubscriberRequest.objects.filter(email__iexact="newuser@example.com").first()
        assert req is not None
        mock_send.assert_called_once()
        req.refresh_from_db()
        assert req.admin_notified_at is not None

    @patch("web.newsserver.auth_helpers._send_email")
    def test_valid_token_existing_subscriber_request_updates_and_notifies_once(
        self, mock_send, client
    ):
        SubscriberRequest.objects.create(
            email="existingreq@example.com",
            first_name="Existing",
            last_name="Request",
        )
        with patch.object(
            settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@newsbot.com"
        ):
            token = create_magic_link_token("existingreq@example.com")
            response = client.get(
                reverse("magic_link_verify", kwargs={"token": token}),
                follow=False,
            )
        assert response.status_code == 302
        assert SubscriberRequest.objects.filter(email__iexact="existingreq@example.com").count() == 1
        mock_send.assert_called_once()


# ----- magic_link_signup_requested -----


@pytest.mark.django_db
class TestMagicLinkSignupRequestedView:
    """Tests for magic_link_signup_requested view."""

    def test_returns_200_and_template(self, client):
        response = client.get(reverse("magic_link_signup_requested"))
        assert response.status_code == 200
        template_names = [t.name for t in response.templates]
        assert "newsserver/subscription_requested.html" in template_names


# ----- _send_email (Resend path) -----


class TestSendEmailResendPath:
    """Tests for _send_email when RESEND_API_KEY is configured."""

    @patch("web.newsserver.auth_helpers.send_via_resend")
    @patch.object(settings, "RESEND_API_KEY", "test-resend-key")
    def test_uses_resend_when_api_key_set(self, mock_send):
        from web.newsserver.auth_helpers import _send_email

        _send_email("to@ex.com", "Subject", "plain text", "<b>html</b>")
        mock_send.assert_called_once()
        _, from_email, to_list, subject = mock_send.call_args[0]
        assert to_list == ["to@ex.com"]
        assert subject == "Subject"

    @patch("web.newsserver.auth_helpers.send_via_resend", side_effect=Exception("fail"))
    @patch.object(settings, "RESEND_API_KEY", "test-resend-key")
    def test_fail_silently_suppresses_resend_error(self, mock_send):
        from web.newsserver.auth_helpers import _send_email

        # should not raise
        _send_email("to@ex.com", "S", "text", fail_silently=True)

    @patch("web.newsserver.auth_helpers.send_via_resend", side_effect=Exception("fail"))
    @patch.object(settings, "RESEND_API_KEY", "test-resend-key")
    def test_reraises_resend_error_when_not_silent(self, mock_send):
        from web.newsserver.auth_helpers import _send_email

        with pytest.raises(Exception, match="fail"):
            _send_email("to@ex.com", "S", "text", fail_silently=False)


# ----- notify_admin_config_suggestion -----


@pytest.mark.django_db
class TestNotifyAdminConfigSuggestion:
    """Tests for notify_admin_config_suggestion."""

    @patch("web.newsserver.auth_helpers._send_email")
    def test_sends_email_with_note(self, mock_send):
        from web.newsserver.auth_helpers import notify_admin_config_suggestion
        from web.newsserver.models import ConfigSuggestion

        with patch.object(settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@ex.com"):
            suggestion = ConfigSuggestion.objects.create(
                name="Tech Digest",
                sources="https://techcrunch.com\nhttps://wired.com",
                note="Please include AI coverage",
                email="user@ex.com",
            )
            notify_admin_config_suggestion(suggestion)

        mock_send.assert_called_once()
        _to, _subject, message = mock_send.call_args[0][:3]
        assert "Please include AI coverage" in message

    @patch("web.newsserver.auth_helpers._send_email")
    def test_sends_email_without_note(self, mock_send):
        from web.newsserver.auth_helpers import notify_admin_config_suggestion
        from web.newsserver.models import ConfigSuggestion

        with patch.object(settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@ex.com"):
            suggestion = ConfigSuggestion.objects.create(
                name="Tech Digest",
                sources="https://techcrunch.com",
                note="",
                email="user@ex.com",
            )
            notify_admin_config_suggestion(suggestion)

        mock_send.assert_called_once()
        _to, _subject, message = mock_send.call_args[0][:3]
        assert "Note:" not in message

    @patch("web.newsserver.auth_helpers._send_email")
    def test_no_op_when_admin_email_not_configured(self, mock_send):
        from web.newsserver.auth_helpers import notify_admin_config_suggestion
        from web.newsserver.models import ConfigSuggestion

        with patch.object(settings, "EMAIL_ADMIN_NOTIFICATION_TO", ""):
            suggestion = ConfigSuggestion.objects.create(
                name="X", sources="https://ex.com", email="u@ex.com",
            )
            notify_admin_config_suggestion(suggestion)

        mock_send.assert_not_called()
