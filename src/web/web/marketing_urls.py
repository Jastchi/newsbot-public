"""
URL configuration for marketing hosts (e.g. the bare site domain).

Activated per-request by ``MarketingHostMiddleware`` when the request
host is in ``settings.MARKETING_HOSTS``. These hosts serve only the
public landing page; the Django app itself lives on
``settings.APP_HOST``.
"""

from django.urls import path

from web.newsserver.views import LandingView

urlpatterns = [
    path("", LandingView.as_view(), name="landing"),
]
