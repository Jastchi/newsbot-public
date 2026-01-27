"""URL configurations for the newsserver app."""

from django.urls import path

from . import views

app_name = "newsserver"

urlpatterns = [
    path("", views.ConfigOverviewView.as_view(), name="overview"),
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
        views.NewsSchedulerDashboardView.as_view(),
        name="news_schedule",
    ),
]
