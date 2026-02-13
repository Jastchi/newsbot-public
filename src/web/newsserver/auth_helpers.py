"""
Helpers and views for magic-link sign-in and sign-up.

Rate limiting, tokens, email, and the magic-link
request/sent/verify/signup views live here.
"""

import secrets
from urllib.parse import quote

from django.conf import settings
from django.contrib.auth import login as auth_login
from django.core.cache import cache
from django.core.mail import send_mail
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone

from .models import Subscriber, SubscriberRequest

# Magic-link: rate limit 2 per window (per IP and per email), in-memory
MAGIC_LINK_RATE_LIMIT_WINDOW_SECONDS = 15 * 60
MAGIC_LINK_RATE_LIMIT_WINDOW_MINUTES = (
    MAGIC_LINK_RATE_LIMIT_WINDOW_SECONDS // 60
)
MAGIC_LINK_RATE_LIMIT_MAX = 2
MAGIC_LINK_TOKEN_TIMEOUT_SECONDS = 15 * 60
MAGIC_LINK_TOKEN_TIMEOUT_MINUTES = MAGIC_LINK_TOKEN_TIMEOUT_SECONDS // 60
CACHE_KEY_MAGIC_LINK_IP = "magic_link_send_ip"
CACHE_KEY_MAGIC_LINK_EMAIL = "magic_link_send_email"
CACHE_KEY_MAGIC_LINK_TOKEN = "magic_link" + "_token"


def get_client_ip(request: HttpRequest) -> str:
    """
    Return client IP for rate limiting.

    Supports X-Forwarded-For when behind a proxy.
    """
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "")


def magic_link_rate_limit_exceeded(request: HttpRequest, email: str) -> bool:
    """
    Check if sending a magic link would exceed rate limit.

    2 per rate-limit window per IP and per email. If not exceeded,
    consume one slot. Returns True if limit exceeded.
    """
    ip = get_client_ip(request)
    email_key = (email or "").strip().lower()

    if not email_key:
        return True

    ip_key = f"{CACHE_KEY_MAGIC_LINK_IP}:{ip}"
    email_limit_key = f"{CACHE_KEY_MAGIC_LINK_EMAIL}:{email_key}"
    count_ip = cache.get(ip_key, 0)
    count_email = cache.get(email_limit_key, 0)

    if (
        count_ip >= MAGIC_LINK_RATE_LIMIT_MAX
        or count_email >= MAGIC_LINK_RATE_LIMIT_MAX
    ):
        return True

    cache.set(ip_key, count_ip + 1, MAGIC_LINK_RATE_LIMIT_WINDOW_SECONDS)
    cache.set(
        email_limit_key,
        count_email + 1,
        MAGIC_LINK_RATE_LIMIT_WINDOW_SECONDS,
    )
    return False


def create_magic_link_token(email: str) -> str:
    """
    Store email under a new token in cache and return the token.

    Token expires after MAGIC_LINK_TOKEN_TIMEOUT_SECONDS.
    """
    token = secrets.token_urlsafe(32)
    cache.set(
        f"{CACHE_KEY_MAGIC_LINK_TOKEN}:{token}",
        email.strip().lower(),
        MAGIC_LINK_TOKEN_TIMEOUT_SECONDS,
    )
    return token


def consume_magic_link_token(token: str) -> str | None:
    """
    Return email for token and delete it from cache.

    Returns None if token missing or expired.
    """
    cache_key = f"{CACHE_KEY_MAGIC_LINK_TOKEN}:{token}"
    email = cache.get(cache_key)

    if email is not None:
        cache.delete(cache_key)

    return email


def build_magic_link_verify_url(
    request: HttpRequest,
    token: str,
    next_url: str = "",
) -> str:
    """Build absolute URL for the magic-link verify view."""
    verify_path = reverse("magic_link_verify", kwargs={"token": token})

    if next_url:
        verify_path += "?next=" + quote(next_url)
    if request.get_host():
        return request.build_absolute_uri(verify_path)

    return verify_path


def send_magic_link_email(email: str, verify_url: str) -> None:
    """Send the magic-link email (login or signup)."""
    subject = "Your login link"
    message = (
        "Click the button below to log in or sign up for the NewsBot "
        f"(valid for {MAGIC_LINK_TOKEN_TIMEOUT_MINUTES} minutes):\n\n"
        f"{verify_url}\n\n"
        "If you did not request this, you can ignore this email."
    )
    html_message = render_to_string(
        "account/magic_link_email.html",
        {
            "verify_url": verify_url,
            "magic_link_valid_minutes": MAGIC_LINK_TOKEN_TIMEOUT_MINUTES,
        },
    )
    send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL or None,
        recipient_list=[email],
        fail_silently=False,
        html_message=html_message,
    )


def notify_admin_subscriber_request(req: SubscriberRequest) -> None:
    """
    Send immediate email to admin when new subscriber request created.

    Only sends once per request (skips if admin_notified_at is already
    set).
    Uses EMAIL_ADMIN_NOTIFICATION_TO; no-op if unset.
    """
    if req.admin_notified_at is not None:
        return
    to = getattr(settings, "EMAIL_ADMIN_NOTIFICATION_TO", "").strip()
    if not to:
        return

    name = f"{req.first_name} {req.last_name}".strip() or "(no name)"
    subject = "NewsBot: New subscription request"
    message = (
        f"A user has requested to be added as a subscriber.\n\n"
        f"Email: {req.email}\n"
        f"Name: {name}\n\n"
        "Process in Django Admin: Subscriber requests → add as Subscriber and "
        "assign configs, or use the Pending requests page."
    )
    send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL or None,
        recipient_list=[to],
        fail_silently=True,
    )
    req.admin_notified_at = timezone.now()
    req.save(update_fields=["admin_notified_at"])


def request_magic_link(request: HttpRequest) -> HttpResponse:
    """
    GET: show form to enter email and request a magic link.

    POST: validate email, apply rate limit, send magic-link email,
    redirect.
    """
    next_url = (
        request.GET.get("next", "").strip()
        or request.POST.get("next", "").strip()
    )
    if request.method == "GET":
        return render(
            request,
            "account/login_by_email.html",
            {
                "next": next_url,
                "magic_link_valid_minutes": MAGIC_LINK_TOKEN_TIMEOUT_MINUTES,
            },
        )
    email = (request.POST.get("email") or "").strip().lower()
    if not email:
        return render(
            request,
            "account/login_by_email.html",
            {
                "next": next_url,
                "error": "Please enter your email address.",
                "magic_link_valid_minutes": MAGIC_LINK_TOKEN_TIMEOUT_MINUTES,
            },
            status=400,
        )
    if magic_link_rate_limit_exceeded(request, email):
        return render(
            request,
            "account/login_by_email.html",
            {
                "next": next_url,
                "error": (
                    f"Too many magic link requests. Try again in "
                    f"{MAGIC_LINK_RATE_LIMIT_WINDOW_MINUTES} minutes."
                ),
                "magic_link_valid_minutes": MAGIC_LINK_TOKEN_TIMEOUT_MINUTES,
            },
            status=429,
        )
    token = create_magic_link_token(email)
    verify_url = build_magic_link_verify_url(request, token, next_url)
    send_magic_link_email(email, verify_url)
    redirect_to = reverse("magic_link_sent")
    if next_url:
        redirect_to += f"?next={next_url}"
    return redirect(redirect_to)


def magic_link_sent(request: HttpRequest) -> HttpResponse:
    """Show 'Check your email' after requesting a magic link."""
    next_url = request.GET.get("next", "").strip()
    return render(
        request,
        "account/magic_link_sent.html",
        {
            "next": next_url,
            "magic_link_valid_minutes": MAGIC_LINK_TOKEN_TIMEOUT_MINUTES,
        },
    )


def verify_magic_link(request: HttpRequest, token: str) -> HttpResponse:
    """
    Verify magic-link token from email.

    If Subscriber exists: log in and redirect.
    If no Subscriber: create SubscriberRequest, notify admin, show
    request received.
    Invalid or expired token: redirect to login with error message.
    """
    email = consume_magic_link_token(token)
    if not email:
        next_url = reverse("account_login")
        return redirect(
            f"{next_url}?error=magic_link_invalid",
        )
    try:
        user = Subscriber.objects.get(email__iexact=email)
        auth_login(request, user, backend=settings.AUTHENTICATION_BACKENDS[0])
        next_url = (
            request.GET.get("next", "").strip() or settings.LOGIN_REDIRECT_URL
        )
        return redirect(next_url)
    except Subscriber.DoesNotExist:
        pass
    obj = SubscriberRequest.objects.filter(email__iexact=email).first()
    if not obj:
        obj = SubscriberRequest.objects.create(email=email)
    notify_admin_subscriber_request(obj)
    return redirect("magic_link_signup_requested")


def magic_link_signup_requested(request: HttpRequest) -> HttpResponse:
    """Show 'Request received' after magic-link signup (no sub)."""
    return render(
        request,
        "newsserver/subscription_requested.html",
    )
