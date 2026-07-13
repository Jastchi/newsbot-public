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
from web.newsserver.views import LandingView, robots_txt

urlpatterns = [
    path("", LandingView.as_view(), name="landing"),
    path("robots.txt", robots_txt, name="robots_txt"),
    path(
        "sitemap.xml",
        sitemap,
        {"sitemaps": {"pages": MarketingSitemap()}},
        name="sitemap",
    ),
]
