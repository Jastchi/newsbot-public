"""
URL configuration for marketing hosts (e.g. the bare site domain).

Activated per-request by ``MarketingHostMiddleware`` when the request
host is in ``settings.MARKETING_HOSTS``. These hosts serve only the
public landing page; the Django app itself lives on
``settings.APP_HOST``.
"""

from django.contrib.sitemaps.views import sitemap
from django.urls import path

from web.newsserver.sitemaps import MarketingSitemap
from web.newsserver.views import (
    LandingView,
    PrivacyPolicyView,
    llms_full_txt,
    llms_txt,
    robots_txt,
    security_txt,
)

urlpatterns = [
    path("", LandingView.as_view(), name="landing"),
    path("privacy/", PrivacyPolicyView.as_view(), name="privacy_policy"),
    path("robots.txt", robots_txt, name="robots_txt"),
    path("llms.txt", llms_txt, name="llms_txt"),
    path("llms-full.txt", llms_full_txt, name="llms_full_txt"),
    path(".well-known/security.txt", security_txt, name="security_txt"),
    path(
        "sitemap.xml",
        sitemap,
        {"sitemaps": {"pages": MarketingSitemap()}},
        name="sitemap",
    ),
]
