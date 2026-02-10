"""Tests for subscriber request and signup (Google-only) views."""

from unittest.mock import patch

import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

from web.newsserver.models import Subscriber, SubscriberRequest

User = get_user_model()


@pytest.mark.django_db
class TestSubscriberRequestCreateView:
    """Tests for POST /news-schedule/request-access/ (subscriber_request_create)."""

    @pytest.fixture
    def url(self):
        return reverse("newsserver:subscriber_request_create")

    def test_post_requires_login(self, client, url):
        """Unauthenticated POST returns redirect to login."""
        response = client.post(url, data={}, content_type="application/json")
        assert response.status_code == 302
        assert "login" in response.url

    def test_post_creates_subscriber_request(self, client, url):
        """Authenticated subscriber with no configs gets a SubscriberRequest created."""
        user = User.objects.create_user(
            email="newuser@example.com",
            first_name="",
            last_name="",
            password="pass",
            is_staff=False,
        )
        client.force_login(user)
        response = client.post(
            url,
            data=b"{}",
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
        req = SubscriberRequest.objects.filter(email="newuser@example.com").first()
        assert req is not None
        assert req.first_name == ""
        assert req.user == user

    def test_post_already_subscriber_returns_400(self, client, url, sample_subscribers):
        """Subscriber who already has configs gets 400."""
        sub = sample_subscribers[0]
        # View treats "has configs" as configs with published_for_subscription=True
        sub.configs.filter(is_active=True).update(published_for_subscription=True)
        client.force_login(sub)
        response = client.post(
            url,
            data=b"{}",
            content_type="application/json",
        )
        assert response.status_code == 400
        assert response.json()["message"] == "Already have subscriptions"
        assert SubscriberRequest.objects.filter(email=sub.email).count() == 0

    def test_post_updates_existing_request(self, client, url):
        """If SubscriberRequest already exists for email, it is updated (no duplicate)."""
        user = User.objects.create_user(
            email="again@example.com",
            first_name="Again",
            last_name="User",
            password="pass",
            is_staff=False,
        )
        existing = SubscriberRequest.objects.create(
            email="again@example.com",
            first_name="Old",
            last_name="Name",
        )
        client.force_login(user)
        response = client.post(
            url,
            data=b"{}",
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert SubscriberRequest.objects.filter(email="again@example.com").count() == 1
        existing.refresh_from_db()
        assert existing.first_name == "Again"
        assert existing.last_name == "User"
        assert existing.user == user

    def test_post_method_not_allowed_get(self, client, url):
        """GET returns 405."""
        user = User.objects.create_user(
            email="u@example.com",
            first_name="",
            last_name="",
            password="pass",
            is_staff=False,
        )
        client.force_login(user)
        response = client.get(url)
        assert response.status_code == 405

    @patch("web.newsserver.auth_helpers.send_mail")
    def test_post_notifies_admin_when_email_configured(
        self, mock_send_mail, client, url
    ):
        """When EMAIL_ADMIN_NOTIFICATION_TO is set, admin is notified and admin_notified_at is set."""
        with patch.object(settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@example.com"):
            with patch.object(
                settings, "DEFAULT_FROM_EMAIL", "newsbot@example.com"
            ):
                user = User.objects.create_user(
                    email="notify@example.com",
                    first_name="Notify",
                    last_name="User",
                    password="pass",
                    is_staff=False,
                )
                client.force_login(user)
                response = client.post(
                    url,
                    data=b"{}",
                    content_type="application/json",
                )
        assert response.status_code == 200
        req = SubscriberRequest.objects.get(email="notify@example.com")
        assert req.admin_notified_at is not None
        mock_send_mail.assert_called_once()
        call_kw = mock_send_mail.call_args[1]
        assert call_kw["recipient_list"] == ["admin@example.com"]
        assert "New subscription request" in call_kw["subject"]
        assert "notify@example.com" in call_kw["message"]

    @patch("web.newsserver.auth_helpers.send_mail")
    def test_post_skips_notify_when_admin_already_notified(
        self, mock_send_mail, client, url
    ):
        """When admin_notified_at is already set, send_mail is not called again."""
        with patch.object(settings, "EMAIL_ADMIN_NOTIFICATION_TO", "admin@example.com"):
            user = User.objects.create_user(
                email="again@example.com",
                first_name="Again",
                last_name="User",
                password="pass",
                is_staff=False,
            )
            SubscriberRequest.objects.create(
                email="again@example.com",
                first_name="Again",
                last_name="User",
                admin_notified_at=timezone.now(),
            )
            client.force_login(user)
            response = client.post(
                url,
                data=b"{}",
                content_type="application/json",
            )
        assert response.status_code == 200
        mock_send_mail.assert_not_called()

    @patch("web.newsserver.auth_helpers.send_mail")
    def test_post_no_notify_when_email_admin_unset(self, mock_send_mail, client, url):
        """When EMAIL_ADMIN_NOTIFICATION_TO is unset, send_mail is not called."""
        with patch.object(settings, "EMAIL_ADMIN_NOTIFICATION_TO", ""):
            user = User.objects.create_user(
                email="noemail@example.com",
                first_name="",
                last_name="",
                password="pass",
                is_staff=False,
            )
            client.force_login(user)
            response = client.post(
                url,
                data=b"{}",
                content_type="application/json",
            )
        assert response.status_code == 200
        mock_send_mail.assert_not_called()


@pytest.mark.django_db
class TestSignupGoogleOnlyView:
    """Tests for signup page (Google-only; no email/password form)."""

    @pytest.fixture
    def url(self):
        return reverse("account_signup")

    def test_get_returns_signup_page(self, client, url):
        """GET returns 200 and signup template with Google option."""
        response = client.get(url)
        assert response.status_code == 200
        assert b"Sign up with Google" in response.content
        assert b"Create an account" in response.content
        assert b"Already have an account" in response.content

    def test_get_passes_redirect_params(self, client, url):
        """GET with next= passes redirect_field_value in context."""
        response = client.get(url, {"next": "/news-schedule/"})
        assert response.status_code == 200
        assert response.context.get("redirect_field_name") == "next"
        assert response.context.get("redirect_field_value") == "/news-schedule/"

    def test_post_returns_405(self, client, url):
        """POST (email/password signup) returns 405."""
        response = client.post(
            url,
            data={
                "email": "someone@example.com",
                "password1": "securepass123",
                "password2": "securepass123",
            },
        )
        assert response.status_code == 405
        assert b"Sign up is only available with Google" in response.content


@pytest.mark.django_db
class TestSubscriptionRequestedView:
    """Tests for news-schedule/requested/ (SubscriptionRequestedView)."""

    @pytest.fixture
    def url(self):
        return reverse("newsserver:subscription_requested")

    def test_get_requires_login(self, client, url):
        """Unauthenticated GET redirects to login."""
        response = client.get(url)
        assert response.status_code == 302
        assert "login" in response.url

    def test_get_returns_requested_page(self, client, url, admin_user):
        """Authenticated GET returns 200 and subscription_requested template."""
        client.force_login(admin_user)
        response = client.get(url)
        assert response.status_code == 200
        assert "newsserver/subscription_requested.html" in [
            t.name for t in response.templates
        ]


@pytest.mark.django_db
class TestSubscriptionRequestFromSocial:
    """Tests for news-schedule/request-from-social/ (subscription_request_from_social)."""

    @pytest.fixture
    def url(self):
        return reverse("newsserver:subscription_request_from_social")

    def test_get_no_session_data_redirects_to_schedule(self, client, url):
        """When session has no subscription request data, redirect to news_schedule."""
        response = client.get(url)
        assert response.status_code == 302
        assert response.url == reverse("newsserver:news_schedule")

    def test_get_session_empty_email_redirects_to_schedule(self, client, url):
        """When session has data but empty email, redirect to news_schedule."""
        session = client.session
        session["newsserver_subscription_request_from_social"] = {
            "email": "  ",
            "first_name": "A",
            "last_name": "B",
        }
        session.save()
        response = client.get(url)
        assert response.status_code == 302
        assert response.url == reverse("newsserver:news_schedule")

    def test_get_session_data_creates_request_and_renders(self, client, url):
        """When session has data, create SubscriberRequest and render requested page."""
        session = client.session
        session["newsserver_subscription_request_from_social"] = {
            "email": "social@example.com",
            "first_name": "Social",
            "last_name": "User",
        }
        session.save()
        response = client.get(url)
        assert response.status_code == 200
        assert "newsserver/subscription_requested.html" in [
            t.name for t in response.templates
        ]
        req = SubscriberRequest.objects.filter(email="social@example.com").first()
        assert req is not None
        assert req.first_name == "Social"
        assert req.last_name == "User"
        # Session key should be popped
        assert "newsserver_subscription_request_from_social" not in client.session

    def test_get_session_data_updates_existing_request(self, client, url):
        """When session has data and request exists, update and render."""
        SubscriberRequest.objects.create(
            email="update@example.com",
            first_name="Old",
            last_name="Name",
        )
        session = client.session
        session["newsserver_subscription_request_from_social"] = {
            "email": "update@example.com",
            "first_name": "New",
            "last_name": "Name",
        }
        session.save()
        response = client.get(url)
        assert response.status_code == 200
        req = SubscriberRequest.objects.get(email="update@example.com")
        assert req.first_name == "New"
        assert req.last_name == "Name"


@pytest.mark.django_db
class TestPendingSubscriptionRequests:
    """Tests for news-schedule/pending-requests/ (staff-only)."""

    @pytest.fixture
    def url(self):
        return reverse("newsserver:pending_subscription_requests")

    def test_get_requires_staff(self, client, url):
        """Non-staff user is redirected or gets 302 to login."""
        user = User.objects.create_user(
            email="nonstaff@example.com",
            first_name="",
            last_name="",
            password="pass",
            is_staff=False,
        )
        client.force_login(user)
        response = client.get(url)
        assert response.status_code in (302, 403)
        if response.status_code == 302:
            assert "login" in response.url

    def test_get_staff_returns_pending_list(self, client, url, admin_user):
        """Staff user gets 200 and pending requests in context."""
        admin_user.is_staff = True
        admin_user.save()
        SubscriberRequest.objects.create(
            email="pending1@example.com",
            first_name="P",
            last_name="One",
        )
        client.force_login(admin_user)
        response = client.get(url)
        assert response.status_code == 200
        assert "pending_requests" in response.context
        assert len(response.context["pending_requests"]) >= 1
        assert "newsserver/pending_subscription_requests.html" in [
            t.name for t in response.templates
        ]


@pytest.mark.django_db
class TestSubscriberSubscriptionsView:
    """Tests for news-schedule/subscriptions/ (subscriber_subscriptions)."""

    @pytest.fixture
    def url(self):
        return reverse("newsserver:subscriber_subscriptions")

    def test_get_requires_login(self, client, url):
        """Unauthenticated GET redirects to login."""
        response = client.get(url)
        assert response.status_code == 302
        assert "login" in response.url

    def test_get_returns_subscribed_config_ids(self, client, url, sample_subscribers, sample_news_configs):
        """GET returns config_ids for published configs the subscriber is subscribed to."""
        sub = sample_subscribers[0]
        config = sample_news_configs[0]
        config.published_for_subscription = True
        config.save()
        sub.configs.add(config)
        client.force_login(sub)
        response = client.get(url)
        assert response.status_code == 200
        data = response.json()
        assert "config_ids" in data
        assert list(data["config_ids"]) == [config.pk]

    def test_get_returns_empty_when_no_subscriptions(self, client, url, sample_subscribers):
        """GET returns empty config_ids when subscriber has no configs."""
        sub = sample_subscribers[0]
        sub.configs.clear()
        client.force_login(sub)
        response = client.get(url)
        assert response.status_code == 200
        assert response.json()["config_ids"] == []

    def test_post_requires_login(self, client, url):
        """Unauthenticated POST redirects to login."""
        response = client.post(
            url,
            data=b"{}",
            content_type="application/json",
        )
        assert response.status_code == 302
        assert "login" in response.url

    def test_post_updates_subscriptions(self, client, url, sample_subscribers, sample_news_configs):
        """POST with valid config_ids updates subscriber configs."""
        sub = sample_subscribers[0]
        config1, config2 = sample_news_configs[0], sample_news_configs[1]
        config1.published_for_subscription = True
        config2.published_for_subscription = True
        config1.save()
        config2.save()
        client.force_login(sub)
        response = client.post(
            url,
            data=b'{"config_ids": [%d, %d]}' % (config1.pk, config2.pk),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert set(data["config_ids"]) == {config1.pk, config2.pk}
        sub.refresh_from_db()
        assert set(sub.configs.values_list("pk", flat=True)) == {config1.pk, config2.pk}

    def test_post_config_ids_not_list_returns_400(self, client, url, sample_subscribers):
        """POST with config_ids not a list returns 400."""
        client.force_login(sample_subscribers[0])
        response = client.post(
            url,
            data=b'{"config_ids": "not-a-list"}',
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "config_ids must be a list" in response.json().get("message", "")

    def test_post_invalid_json_returns_400(self, client, url, sample_subscribers):
        """POST with invalid JSON returns 400."""
        client.force_login(sample_subscribers[0])
        response = client.post(
            url,
            data=b"not json",
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "message" in response.json()

    def test_post_method_not_allowed_for_put(self, client, url, sample_subscribers):
        """PUT or other method returns 405."""
        client.force_login(sample_subscribers[0])
        response = client.put(
            url,
            data=b"{}",
            content_type="application/json",
        )
        assert response.status_code == 405
