"""URL configurations for the newsserver app."""

from django.urls import path, reverse_lazy
from django.views.generic import RedirectView

from . import views

app_name = "newsserver"

urlpatterns = [
    path("", views.NewsSchedulerDashboardView.as_view(), name="news_schedule"),
    path(
        "report-archive/",
        views.ConfigOverviewView.as_view(),
        name="overview",
    ),
    path("runs/", views.RunListView.as_view(), name="runs_list"),
    path("logs/", views.LogsView.as_view(), name="logs_list"),
    path("logs/stream/", views.log_stream_view, name="logs_stream"),
    path(
        "config/<str:config_name>/",
        views.ConfigReportView.as_view(),
        name="config_report",
    ),
    path(
        "news-schedule/",
        RedirectView.as_view(
            url=reverse_lazy("newsserver:news_schedule"), permanent=False,
        ),
    ),
    path(
        "news-schedule/requested/",
        views.SubscriptionRequestedView.as_view(),
        name="subscription_requested",
    ),
    path(
        "news-schedule/request-from-social/",
        views.subscription_request_from_social,
        name="subscription_request_from_social",
    ),
    path(
        "news-schedule/pending-requests/",
        views.pending_subscription_requests,
        name="pending_subscription_requests",
    ),
    path(
        "news-schedule/request-access/",
        views.subscriber_request_create,
        name="subscriber_request_create",
    ),
    path(
        "news-schedule/subscriptions/",
        views.subscriber_subscriptions,
        name="subscriber_subscriptions",
    ),
]
