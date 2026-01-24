"""Screenshot tests for the Django web interface.

This module captures screenshots of all tabs in the web interface
for visual regression testing and documentation purposes.
"""

import json

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

    return {
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


@pytest.mark.django_db(transaction=True)
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
        from playwright.sync_api import sync_playwright

        base_url = django_live_server["url"]
        sample_data = django_live_server["sample_data"]

        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()

            # ==========================================
            # Main Interface Screenshots
            # ==========================================

            # 1. Configs Overview page
            page.goto(f"{base_url}/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "configs_overview.png"))

            # 2. Runs List page
            page.goto(f"{base_url}/runs/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "runs_list.png"))

            # 3. Logs page (default view)
            page.goto(f"{base_url}/logs/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "logs_default.png"))

            # 4. Logs page with each config tab
            for config in sample_data["news_configs"]:
                log_url = f"{base_url}/logs/?log={config.key}.log"
                page.goto(log_url)
                page.wait_for_load_state("networkidle")
                page.screenshot(
                    path=str(screenshot_dir / f"logs_{config.key}.png")
                )

            # 5. News Schedule page (Calendar view)
            page.goto(f"{base_url}/news-schedule/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "news_scheduler_calendar.png"))

            # ==========================================
            # Admin Interface Screenshots
            # ==========================================

            # Login to admin
            page.goto(f"{base_url}/admin/login/")
            page.wait_for_load_state("networkidle")
            page.fill('input[name="username"]', "admin")
            page.fill('input[name="password"]', "admin123")
            page.click('input[type="submit"]')
            page.wait_for_load_state("networkidle")
            
            # Wait for redirect to admin index (login successful)
            page.wait_for_url(f"{base_url}/admin/**", timeout=5000)

            # Admin Index (already navigated after login)
            page.goto(f"{base_url}/admin/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "admin_index.png"))

            # NewsConfig List
            page.goto(f"{base_url}/admin/newsserver/newsconfig/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "admin_newsconfig_list.png"))

            # NewsConfig Detail (for each config)
            for config in sample_data["news_configs"]:
                page.goto(
                    f"{base_url}/admin/newsserver/newsconfig/{config.id}/change/"
                )
                page.wait_for_load_state("networkidle")
                page.screenshot(
                    path=str(
                        screenshot_dir / f"admin_newsconfig_detail_{config.key}.png"
                    )
                )

            # Subscriber List
            page.goto(f"{base_url}/admin/newsserver/subscriber/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "admin_subscriber_list.png"))

            # Subscriber Detail (for first subscriber)
            if sample_data["subscribers"]:
                subscriber = sample_data["subscribers"][0]
                page.goto(
                    f"{base_url}/admin/newsserver/subscriber/{subscriber.id}/change/"
                )
                page.wait_for_load_state("networkidle")
                page.screenshot(
                    path=str(screenshot_dir / "admin_subscriber_detail.png")
                )

            # Schedule as admin
            page.goto(f"{base_url}/news-schedule/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "admin_news_scheduler_calendar.png"))

            # Article List
            page.goto(f"{base_url}/admin/newsserver/article/")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=str(screenshot_dir / "admin_article_list.png"))

            # Article Detail (for first article)
            if sample_data["articles"]:
                article = sample_data["articles"][0]
                page.goto(
                    f"{base_url}/admin/newsserver/article/{article.id}/change/"
                )
                page.wait_for_load_state("networkidle")
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

        # Verify screenshots were created
        expected_screenshots = [
            "configs_overview.png",
            "runs_list.png",
            "logs_default.png",
            "news_scheduler_calendar.png",
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
