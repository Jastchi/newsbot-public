"""Signal handlers for the newsserver app."""

import logging

from django.conf import settings
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import send_mail
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse

from .models import Subscriber, SubscriberRequest

logger = logging.getLogger(__name__)


def _build_login_url() -> str:
    """Build absolute URL for the login page (for emails)."""
    site = get_current_site(None)
    debug = getattr(settings, "DEBUG", True)
    protocol = "https" if not debug else "http"
    script = getattr(settings, "FORCE_SCRIPT_NAME", "") or ""
    path = reverse("account_login")
    return f"{protocol}://{site.domain}{script}{path}"


@receiver(post_save, sender=Subscriber)
def notify_user_when_subscriber_request_accepted(
    _sender: type[Subscriber],
    instance: Subscriber,
    *,
    created: bool,
    **_kwargs: object,
) -> None:
    """
    Notify user when their subscription request is accepted.

    When a Subscriber is created and a SubscriberRequest exists for that
    email, send the user an email that their request has been accepted.
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
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL or None,
            recipient_list=[email],
            fail_silently=True,
        )
    except Exception as e:
        logger.warning(
            "Failed to send subscription-accepted email to %s: %s",
            email,
            e,
        )
