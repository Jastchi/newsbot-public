"""Screenshot tests for the Django web interface.

This module captures screenshots of all tabs in the web interface
for visual regression testing and documentation purposes.

Requires Playwright browsers: run `playwright install`.
"""

import json
import os
from typing import Any, Sequence, cast

import pytest


def serialize_mock_data(sample_data: dict) -> dict:
    """
    Serialize all mock data to a dictionary for JSON export.

    Args:
        sample_data: Dictionary containing all sample data objects

    Returns:
        Dictionary with serialized mock data

    """
    def serialize_config(config):
        return {
            "key": config.key,
            "display_name": config.display_name,
            "country": config.country,
            "language": config.language,
            "is_active": config.is_active,
            "published_for_subscription": getattr(
                config, "published_for_subscription", False
            ),
        }

    def serialize_source(source):
        return {
            "name": source.name,
            "url": source.url,
            "type": source.type,
        }

    def serialize_article(article):
        return {
            "config_file": article.config_file,
            "title": article.title,
            "source": article.source,
            "url": article.url,
            "sentiment_label": article.sentiment_label,
            "sentiment_score": article.sentiment_score,
        }

    def serialize_scrape_summary(summary):
        return {
            "config": summary.config.key if summary.config else None,
            "success": summary.success,
            "duration": summary.duration,
            "articles_scraped": summary.articles_scraped,
            "articles_saved": summary.articles_saved,
        }

    def serialize_analysis_summary(summary):
        return {
            "config": summary.config.key if summary.config else None,
            "success": summary.success,
            "duration": summary.duration,
            "articles_analyzed": summary.articles_analyzed,
            "stories_identified": summary.stories_identified,
        }

    def serialize_subscriber(subscriber):
        return {
            "first_name": subscriber.first_name,
            "last_name": subscriber.last_name,
            "email": subscriber.email,
            "is_active": subscriber.is_active,
            "configs": [c.key for c in subscriber.configs.all()],
        }

    def serialize_subscriber_request(req):
        return {
            "email": req.email,
            "first_name": req.first_name,
            "last_name": req.last_name,
        }

    result = {
        "news_configs": [serialize_config(c) for c in sample_data["news_configs"]],
        "news_sources": [serialize_source(s) for s in sample_data["news_sources"]],
        "articles": [serialize_article(a) for a in sample_data["articles"]],
        "scrape_summaries": [
            serialize_scrape_summary(s) for s in sample_data["scrape_summaries"]
        ],
        "analysis_summaries": [
            serialize_analysis_summary(s) for s in sample_data["analysis_summaries"]
        ],
        "subscribers": [serialize_subscriber(s) for s in sample_data["subscribers"]],
    }
    if "subscriber_requests" in sample_data:
        result["subscriber_requests"] = [
            serialize_subscriber_request(r) for r in sample_data["subscriber_requests"]
        ]
    return result


# available_apps enables TRUNCATE ... CASCADE on teardown (PostgreSQL)
_SCREENSHOT_AVAILABLE_APPS = (
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
    "web.newsserver",
)


@pytest.mark.django_db(transaction=True, available_apps=_SCREENSHOT_AVAILABLE_APPS)
class TestWebInterfaceScreenshots:
    """Test suite for capturing screenshots of all web interface tabs."""

    def test_screenshot_all_tabs(self, django_live_server, screenshot_dir):
        """
        Capture screenshots of all navigation tabs and admin pages.

        This test:
        1. Starts a live Django server with sample data
        2. Uses Playwright to navigate to each tab
        3. Captures screenshots of each page
        4. Saves mock data as JSON for comparison
        """
        from django.conf import settings as django_settings
        from django.contrib.sessions.backends.db import SessionStore
        from playwright.sync_api import sync_playwright

        base_url = django_live_server["url"]
        sample_data = django_live_server["sample_data"]
        admin_user = sample_data["admin_user"]

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as e:
                err = str(e).lower()
                if (
                    "executable doesn't exist" in err
                    or "playwright install" in err
                    or "browser not found" in err
                ):
                    pytest.skip(
                        "Playwright browsers not installed. Run: playwright install"
                    )
                if "target" in err and "closed" in err:
                    pytest.skip(
                        "Browser crashed or closed (e.g. SIGSEGV in sandbox). "
                        "Run outside sandbox or ensure Playwright browsers are installed."
                    )
                raise

            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
            )
            context.set_default_timeout(30_000)
            page = context.new_page()

            # Log in as admin via session (all main views require login).
            # Establish origin first, then set session cookie by URL so it's sent on all requests.
            # Allow synchronous DB access here – the live_server fixture starts an
            # async event loop in this thread, but session creation is a simple ORM call.
            _prev = os.environ.get("DJANGO_ALLOW_ASYNC_UNSAFE")
            os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
            try:
                backend = django_settings.AUTHENTICATION_BACKENDS[0]
                session = SessionStore()
                session["_auth_user_id"] = str(admin_user.pk)
                session["_auth_user_backend"] = backend
                session["_auth_user_hash"] = admin_user.get_session_auth_hash()
                session.save()
            finally:
                if _prev is None:
                    os.environ.pop("DJANGO_ALLOW_ASYNC_UNSAFE", None)
                else:
                    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = _prev
            session_key = session.session_key
            if session_key is None:
                raise RuntimeError("Session key not set after save")

            # Navigate to site first so we can set cookie for this origin
            page.goto(base_url)
            page.wait_for_load_state("load")
            # Set session cookie by URL (most reliable for same-origin)
            session_cookie = {
                "name": django_settings.SESSION_COOKIE_NAME,
                "value": session_key,
                "url": base_url,
            }
            context.add_cookies(
                cast(Sequence[Any], [session_cookie])
            )

            # ==========================================
            # Main Interface Screenshots (match nav tabs in base.html)
            # ==========================================

            # 1. Schedule (root tab) – now logged in; wait for FullCalendar events fetch
            def _is_calendar_events(r):
                u = r.request.url
                return "ajax=1" in u and "start=" in u

            with page.expect_response(_is_calendar_events, timeout=15_000) as response_info:
                page.goto(f"{base_url}/")
                page.wait_for_load_state("load")
                response_info.value  # block until events response received
            if "/accounts/login/" in page.url or "/admin/login/" in page.url:
                raise AssertionError(
                    "Session cookie did not log in; redirected to login. "
                    "Live server and test must share the same database."
                )
            page.wait_for_selector(".fc-event", state="visible", timeout=5_000)
            page.screenshot(path=str(screenshot_dir / "schedule.png"))

            # 2. Report archive tab
            page.goto(f"{base_url}/report-archive/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "report_archive.png"))

            # 3. Runs tab
            page.goto(f"{base_url}/runs/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "runs_list.png"))

            # 4. Logs tab (default view)
            page.goto(f"{base_url}/logs/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "logs_default.png"))

            # 5. Logs page with each config tab
            for config in sample_data["news_configs"]:
                log_url = f"{base_url}/logs/?log={config.key}.log"
                page.goto(log_url)
                page.wait_for_load_state("load")
                page.screenshot(
                    path=str(screenshot_dir / f"logs_{config.key}.png")
                )

            # 6. Pending requests tab (staff only; we are logged in as admin)
            page.goto(f"{base_url}/news-schedule/pending-requests/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "pending_requests.png"))

            # ==========================================
            # Admin Interface Screenshots
            # ==========================================
            # Admin session was set via cookie above; go directly to admin index

            # Admin Index (session cookie set above; no form login needed)
            page.goto(f"{base_url}/admin/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "admin_index.png"))

            # NewsConfig List
            page.goto(f"{base_url}/admin/newsserver/newsconfig/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "admin_newsconfig_list.png"))

            # NewsConfig Detail (for each config)
            for config in sample_data["news_configs"]:
                page.goto(
                    f"{base_url}/admin/newsserver/newsconfig/{config.id}/change/"
                )
                page.wait_for_load_state("load")
                page.screenshot(
                    path=str(
                        screenshot_dir / f"admin_newsconfig_detail_{config.key}.png"
                    )
                )

            # Subscriber List
            page.goto(f"{base_url}/admin/newsserver/subscriber/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "admin_subscriber_list.png"))

            # Subscriber Detail (for first subscriber)
            if sample_data["subscribers"]:
                subscriber = sample_data["subscribers"][0]
                page.goto(
                    f"{base_url}/admin/newsserver/subscriber/{subscriber.id}/change/"
                )
                page.wait_for_load_state("load")
                page.screenshot(
                    path=str(screenshot_dir / "admin_subscriber_detail.png")
                )

            # Schedule as admin (same as main Schedule tab)
            with page.expect_response(_is_calendar_events, timeout=15_000) as response_info:
                page.goto(f"{base_url}/")
                page.wait_for_load_state("load")
                response_info.value
            page.wait_for_selector(".fc-event", state="visible", timeout=5_000)
            page.screenshot(path=str(screenshot_dir / "admin_schedule.png"))

            # Article List
            page.goto(f"{base_url}/admin/newsserver/article/")
            page.wait_for_load_state("load")
            page.screenshot(path=str(screenshot_dir / "admin_article_list.png"))

            # Article Detail (for first article)
            if sample_data["articles"]:
                article = sample_data["articles"][0]
                page.goto(
                    f"{base_url}/admin/newsserver/article/{article.id}/change/"
                )
                page.wait_for_load_state("load")
                page.screenshot(path=str(screenshot_dir / "admin_article_detail.png"))

            # Close browser
            context.close()
            browser.close()

        # ==========================================
        # Save Mock Data as JSON
        # ==========================================

        mock_data = serialize_mock_data(sample_data)
        mock_data_path = screenshot_dir / "mock_data.json"
        with open(mock_data_path, "w") as f:
            json.dump(mock_data, f, indent=2, default=str)

        # Verify screenshots were created (match current nav tabs + admin)
        expected_screenshots = [
            "schedule.png",
            "report_archive.png",
            "runs_list.png",
            "logs_default.png",
            "pending_requests.png",
            "admin_index.png",
            "admin_newsconfig_list.png",
            "admin_subscriber_list.png",
            "admin_article_list.png",
        ]

        for screenshot in expected_screenshots:
            screenshot_path = screenshot_dir / screenshot
            assert screenshot_path.exists(), f"Screenshot {screenshot} was not created"

        # Verify mock data was created
        assert mock_data_path.exists(), "mock_data.json was not created"
