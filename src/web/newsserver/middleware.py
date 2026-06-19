"""HTTP middleware for the newsserver app."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from django.conf import settings
from django.http import HttpResponsePermanentRedirect

from .site_urls import raw_request_host, request_host

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.http import HttpRequest, HttpResponse

    class _URLConfRequest(HttpRequest):
        """HttpRequest subtype with ``urlconf`` override."""

        urlconf: str


class CanonicalHostMiddleware:
    """
    Redirect non-canonical hostnames to the configured site domain.

    Uses 301 so search engines consolidate on one hostname.
    """

    def __init__(
        self,
        get_response: Callable[[HttpRequest], HttpResponse],
    ) -> None:
        """Store the next middleware or view callable."""
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Redirect alternate hosts or pass the request through."""
        canonical_host = settings.SITE_DOMAIN
        if not canonical_host:
            return self.get_response(request)

        host = raw_request_host(request)
        if host == canonical_host:
            return self.get_response(request)

        # Never redirect the app subdomain or marketing hosts: in a
        # subdomain split these are legitimate destinations, not aliases
        # of the canonical host.
        if host == settings.APP_HOST or host in settings.MARKETING_HOSTS:
            return self.get_response(request)

        alternate_hosts = {f"www.{canonical_host}"}
        alternate_hosts.update(
            normalize_host(allowed_host)
            for allowed_host in settings.ALLOWED_HOSTS
            if allowed_host and allowed_host not in {"localhost", "127.0.0.1"}
        )
        if host not in alternate_hosts:
            return self.get_response(request)

        target = f"{settings.CANONICAL_SITE_URL}{request.get_full_path()}"
        return HttpResponsePermanentRedirect(target)


def normalize_host(host: str) -> str:
    """Lowercase hostname without port."""
    return host.split(":", 1)[0].lower()


class MarketingHostMiddleware:
    """
    Serve the public landing page on marketing hosts.

    When ``settings.MARKETING_HOSTS`` is configured and the request
    host matches one, the request is routed through the marketing
    URLconf (the landing page only) instead of ``ROOT_URLCONF`` (the
    full app). With no marketing hosts configured this is a no-op, so
    single-host deployments are unaffected.
    """

    def __init__(
        self,
        get_response: Callable[[HttpRequest], HttpResponse],
    ) -> None:
        """Store the next middleware or view callable."""
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Switch to the marketing URLconf for marketing hosts."""
        if settings.MARKETING_HOSTS and (
            request_host(request) in settings.MARKETING_HOSTS
        ):
            cast("_URLConfRequest", request).urlconf = "web.web.marketing_urls"
        return self.get_response(request)
