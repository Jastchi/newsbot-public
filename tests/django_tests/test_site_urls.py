"""Tests for canonical site URL helpers and middleware."""

import pytest
from django.test import Client, override_settings


@pytest.mark.django_db
class TestCanonicalHostMiddleware:
    """301 redirects from alternate hostnames to the canonical domain."""

    @override_settings(
        SITE_DOMAIN="thenewsbot.net",
        CANONICAL_SITE_URL="https://thenewsbot.net",
        ALLOWED_HOSTS=["thenewsbot.net", "www.thenewsbot.net", "testserver"],
    )
    def test_www_redirects_to_canonical(self):
        client = Client()
        response = client.get("/", HTTP_HOST="www.thenewsbot.net")
        assert response.status_code == 301
        assert response["Location"] == "https://thenewsbot.net/"

    @override_settings(
        SITE_DOMAIN="thenewsbot.net",
        CANONICAL_SITE_URL="https://thenewsbot.net",
        ALLOWED_HOSTS=["thenewsbot.net", "www.thenewsbot.net", "testserver"],
    )
    def test_canonical_host_not_redirected(self):
        client = Client()
        response = client.get("/", HTTP_HOST="thenewsbot.net")
        assert response.status_code == 200

    @override_settings(
        SITE_DOMAIN="",
        CANONICAL_SITE_URL="",
        ALLOWED_HOSTS=["testserver", "www.example.com"],
    )
    def test_no_redirect_when_site_domain_unset(self):
        client = Client()
        response = client.get("/", HTTP_HOST="www.example.com")
        assert response.status_code == 200


@pytest.mark.django_db
class TestCanonicalUrlsContext:
    """Canonical URL exposed in page meta tags."""

    @override_settings(
        SITE_DOMAIN="thenewsbot.net",
        CANONICAL_SITE_URL="https://thenewsbot.net",
        ALLOWED_HOSTS=["thenewsbot.net", "www.thenewsbot.net", "testserver"],
    )
    def test_homepage_includes_canonical_link(self):
        client = Client()
        response = client.get("/", HTTP_HOST="www.thenewsbot.net")
        assert response.status_code == 301

        response = client.get("/", HTTP_HOST="thenewsbot.net")
        content = response.content.decode()
        assert '<link rel="canonical" href="https://thenewsbot.net/">' in content
        assert 'property="og:url" content="https://thenewsbot.net/"' in content


class TestSiteUrlHelpers:
    """Unit tests for site_urls helpers."""

    def test_normalize_site_domain_strips_scheme_and_www(self):
        from web.newsserver.site_urls import normalize_site_domain

        assert normalize_site_domain("https://www.thenewsbot.net/") == "thenewsbot.net"
        assert normalize_site_domain("http://thenewsbot.net") == "thenewsbot.net"

    def test_canonical_site_url_from_env(self):
        from web.newsserver.site_urls import canonical_site_url_from_env

        assert (
            canonical_site_url_from_env(site_domain="www.thenewsbot.net")
            == "https://thenewsbot.net"
        )
        assert (
            canonical_site_url_from_env(news_server_base_url="https://thenewsbot.net")
            == "https://thenewsbot.net"
        )

    @override_settings(
        SITE_DOMAIN="thenewsbot.net",
        CANONICAL_SITE_URL="https://thenewsbot.net",
        FORCE_SCRIPT_NAME="",
    )
    def test_build_canonical_absolute_uri(self, rf):
        from web.newsserver.site_urls import build_canonical_absolute_uri

        request = rf.get("/")
        request.get_host = lambda: "www.thenewsbot.net"
        url = build_canonical_absolute_uri(
            request,
            "/accounts/login-by-email/verify/abc123/",
        )
        assert url == (
            "https://thenewsbot.net/accounts/login-by-email/verify/abc123/"
        )
