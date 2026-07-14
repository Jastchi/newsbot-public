"""Sitemap for the public marketing host (see marketing_urls.py)."""

from urllib.parse import urlsplit

from django.conf import settings
from django.contrib.sitemaps import Sitemap
from django.contrib.sites.models import Site
from django.contrib.sites.requests import RequestSite
from django.urls import reverse


class MarketingSitemap(Sitemap):
    """The indexable URLs on marketing hosts: landing and privacy."""

    def items(self) -> list[str]:
        """Return the URL names to include in the sitemap."""
        return ["landing", "privacy_policy"]

    def location(self, item: str) -> str:
        """Resolve a URL name to its path."""
        return reverse(item)

    def changefreq(self, item: str) -> str:
        """Landing page changes often; the privacy page rarely does."""
        return "weekly" if item == "landing" else "yearly"

    def priority(self, item: str) -> float:
        """Landing page is the primary page; privacy is secondary."""
        return 1.0 if item == "landing" else 0.3

    def get_domain(self, site: Site | RequestSite | None = None) -> str:
        """
        Use CANONICAL_SITE_URL instead of the sites framework.

        The ``Site`` row for SITE_ID isn't kept in sync with the real
        domain (it's only installed here for allauth), so the default
        lookup would emit the framework's default "example.com" domain
        instead of the real one.
        """
        if settings.CANONICAL_SITE_URL:
            return urlsplit(settings.CANONICAL_SITE_URL).netloc
        return super().get_domain(site)

    def get_protocol(self, protocol: str | None = None) -> str:
        """Use CANONICAL_SITE_URL's scheme when configured."""
        if settings.CANONICAL_SITE_URL:
            return urlsplit(settings.CANONICAL_SITE_URL).scheme
        return super().get_protocol(protocol)
