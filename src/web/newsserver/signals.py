"""Signal handlers for the newsserver app."""

import logging

from django.conf import settings
from django.contrib.sites.shortcuts import get_current_site
from django.db.models.signals import post_save
from django.urls import reverse

from .auth_helpers import _send_email
from .models import Subscriber, SubscriberRequest

logger = logging.getLogger(__name__)

DISPATCH_UID_SUBSCRIBER_ACCEPTED = "newsserver.subscriber_request_accepted"


def _build_login_url() -> str:
    """
    Build absolute URL for the login page (for emails).

    In a subdomain split (``APP_HOST`` set) the login page only exists
    on the app host — marketing hosts serve the landing-page URLconf,
    where this path 404s. So prefer ``APP_BASE_URL``, then the
    canonical site, mirroring ``build_magic_link_verify_url``. The
    ``django.contrib.sites`` domain is the last resort: with no
    ``SITE_DOMAIN``/``APP_HOST`` configured it is Django's default
    ``example.com`` unless the row was edited by hand.
    """
    path = reverse("account_login")
    if settings.APP_BASE_URL:
        return f"{settings.APP_BASE_URL}{path}"
    if settings.CANONICAL_SITE_URL:
        return f"{settings.CANONICAL_SITE_URL}{path}"
    protocol = "http" if settings.DEBUG else "https"
    return f"{protocol}://{get_current_site(None).domain}{path}"


def notify_user_when_subscriber_request_accepted(
    instance: Subscriber,
    *,
    created: bool,
    **_kwargs: object,
) -> None:
    """
    Notify user when their subscription request is accepted.

    When a Subscriber is created and a SubscriberRequest exists for that
    email, send the user an email that their request has been accepted.

    ``sender``, ``signal`` and the rest of post_save's keyword arguments
    are absorbed by ``**_kwargs``: the dispatcher passes every argument
    by keyword, so a positional ``sender`` parameter would never bind.
    """
    if not created:
        return
    email = (getattr(instance, "email", None) or "").strip().lower()
    if not email:
        return
    if not SubscriberRequest.objects.filter(email__iexact=email).exists():
        return
    login_url = _build_login_url()
    subject = "Your subscription request has been accepted"
    msg_login = (
        f"You can now log in and manage your subscriptions here:\n{login_url}"
    )
    message = (
        f"Hello,\n\n"
        f"Your request to be added as a subscriber has been accepted.\n\n"
        f"{msg_login}\n\n"
        f"If you have any questions, please contact the administrator."
    )
    try:
        _send_email(email, subject, message, fail_silently=True)
    except Exception as e:
        logger.warning(
            "Failed to send subscription-accepted email to %s: %s",
            email,
            e,
        )


def connect_signal_handlers() -> None:
    """
    Register this module's receivers. Called from ``AppConfig.ready``.

    ``dispatch_uid`` keeps a second ``ready()`` (e.g. under the test
    runner) from connecting the handler twice.
    """
    post_save.connect(
        notify_user_when_subscriber_request_accepted,
        sender=Subscriber,
        dispatch_uid=DISPATCH_UID_SUBSCRIBER_ACCEPTED,
    )
