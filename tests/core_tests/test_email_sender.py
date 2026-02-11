"""Tests for email sender helper functions."""

import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestGetAvailableNewsletters:
    """Test cases for get_available_newsletters function."""

    @pytest.mark.django_db
    def test_get_available_newsletters_from_database(self):
        """Test that get_available_newsletters queries database correctly."""
        from after_analysis.email_sender import get_available_newsletters

        # Mock Django setup and NewsConfig model
        # get_available_newsletters uses .filter(is_active=True).order_by().values_list()
        mock_queryset = MagicMock()
        mock_queryset.filter.return_value.order_by.return_value.values_list.return_value = [
            "Sports Updates",
            "Technology News",
            "World News",
        ]

        with patch(
            "after_analysis.email_sender.NewsConfig.objects",
            mock_queryset,
        ):
            newsletters = get_available_newsletters()

            assert len(newsletters) == 3
            assert "Sports Updates" in newsletters
            assert "Technology News" in newsletters
            assert "World News" in newsletters
            # Should be sorted alphabetically
            assert newsletters == [
                "Sports Updates",
                "Technology News",
                "World News",
            ]

    @pytest.mark.django_db
    def test_get_available_newsletters_empty_database(self):
        """Test get_available_newsletters with empty database."""
        from after_analysis.email_sender import get_available_newsletters

        mock_queryset = MagicMock()
        mock_queryset.filter.return_value.order_by.return_value.values_list.return_value = []

        with patch(
            "after_analysis.email_sender.NewsConfig.objects",
            mock_queryset,
        ):
            newsletters = get_available_newsletters()
            assert newsletters == []

    @pytest.mark.django_db
    def test_get_available_newsletters_database_error(self):
        """Test get_available_newsletters handles database errors gracefully."""
        from after_analysis.email_sender import get_available_newsletters

        with patch(
            "after_analysis.email_sender.NewsConfig.objects",
            side_effect=Exception("Database error"),
        ):
            newsletters = get_available_newsletters()
            assert newsletters == []


class TestReplacePlaceholdersInReport:
    """Test cases for replace_placeholders_in_report function."""

    def test_replace_email_placeholder(self, tmp_path):
        """Test that email placeholder is replaced correctly."""
        from after_analysis.email_sender import replace_placeholders_in_report

        report_html = """
        <html>
            <body>
                <p>Contact us at PLACEHOLDER_EMAIL_ADDRESS</p>
            </body>
        </html>
        """

        with patch(
            "after_analysis.email_sender.get_available_newsletters",
            return_value=["Tech News", "Sports"],
        ):
            result = replace_placeholders_in_report(
                report_html,
                "test@example.com",
            )

            assert "PLACEHOLDER_EMAIL_ADDRESS" not in result
            assert "test@example.com" in result

    def test_replace_newsletters_placeholder(self, tmp_path):
        """Test that newsletters placeholder is replaced correctly."""
        from after_analysis.email_sender import replace_placeholders_in_report

        report_html = """
        <html>
            <body>
                <p>Available newsletters: PLACEHOLDER_NEWSLETTERS</p>
            </body>
        </html>
        """

        with patch(
            "after_analysis.email_sender.get_available_newsletters",
            return_value=["Tech News", "Sports", "World News"],
        ):
            result = replace_placeholders_in_report(
                report_html,
                "test@example.com",
            )

            assert "PLACEHOLDER_NEWSLETTERS" not in result
            assert "Tech News, Sports, World News." in result

    def test_replace_newsletters_placeholder_empty_list(self, tmp_path):
        """Test newsletters placeholder with empty newsletter list."""
        from after_analysis.email_sender import replace_placeholders_in_report

        report_html = """
        <html>
            <body>
                <p>Available newsletters: PLACEHOLDER_NEWSLETTERS</p>
            </body>
        </html>
        """

        with patch(
            "after_analysis.email_sender.get_available_newsletters",
            return_value=[],
        ):
            result = replace_placeholders_in_report(
                report_html,
                "test@example.com",
            )

            assert "PLACEHOLDER_NEWSLETTERS" not in result
            # Should replace with empty string when no newsletters
            assert "Available newsletters: </p>" in result

    def test_replace_both_placeholders(self, tmp_path):
        """Test that both placeholders are replaced correctly."""
        from after_analysis.email_sender import replace_placeholders_in_report

        report_html = """
        <html>
            <body>
                <p>Contact: PLACEHOLDER_EMAIL_ADDRESS</p>
                <p>Newsletters: PLACEHOLDER_NEWSLETTERS</p>
            </body>
        </html>
        """

        with patch(
            "after_analysis.email_sender.get_available_newsletters",
            return_value=["Newsletter A", "Newsletter B"],
        ):
            result = replace_placeholders_in_report(
                report_html,
                "sender@example.com",
            )

            assert "PLACEHOLDER_EMAIL_ADDRESS" not in result
            assert "PLACEHOLDER_NEWSLETTERS" not in result
            assert "sender@example.com" in result
            assert "Newsletter A, Newsletter B." in result

    def test_replace_manage_subscriptions_link_with_base_url(self):
        """When NEWSSERVER_BASE_URL is set, PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK becomes a link."""
        from after_analysis.email_sender import replace_placeholders_in_report

        report_html = "<p>To cancel, PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK.</p>"
        with patch.dict(
            "os.environ",
            {"NEWSSERVER_BASE_URL": "https://news.example.com"},
            clear=False,
        ):
            result = replace_placeholders_in_report(report_html, "bot@example.com")
        assert "PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK" not in result
        assert 'href="https://news.example.com"' in result
        assert "click here" in result

    def test_replace_manage_subscriptions_link_without_base_url(self):
        """When NEWSSERVER_BASE_URL is not set, placeholder becomes contact fallback."""
        from after_analysis.email_sender import replace_placeholders_in_report

        report_html = "<p>To cancel, PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK.</p>"
        with patch.dict("os.environ", {}, clear=False):
            # Remove in case it's set in environment
            os.environ.pop("NEWSSERVER_BASE_URL", None)
            result = replace_placeholders_in_report(report_html, "support@example.com")
        assert "PLACEHOLDER_MANAGE_SUBSCRIPTIONS_LINK" not in result
        assert "contact us at" in result
        assert "support@example.com" in result


class TestEmailEnabledCheck:
    """Test cases for EMAIL_ENABLED environment variable check."""

    def test_execute_disabled_when_email_enabled_false(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Test that execute returns early when EMAIL_ENABLED is false."""
        from after_analysis.email_sender import execute
        from newsbot.models import AnalysisData

        monkeypatch.setenv("EMAIL_ENABLED", "false")

        report_path = tmp_path / "test_report.html"
        report_path.write_text("<html><body>Test</body></html>")

        analysis_data: AnalysisData = {
            "config_name": "test",
            "config_key": "test",
            "stories_count": 5,
            "from_date": datetime(2025, 1, 1),
            "to_date": datetime(2025, 1, 2),
        }

        # Should return early without errors
        with patch("smtplib.SMTP") as smtp_mock:
            execute(report_path, analysis_data)
            smtp_mock.assert_not_called()

    def test_execute_disabled_when_email_enabled_not_set(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Test that execute returns early when EMAIL_ENABLED is not set."""
        from after_analysis.email_sender import execute
        from newsbot.models import AnalysisData

        monkeypatch.delenv("EMAIL_ENABLED", raising=False)

        report_path = tmp_path / "test_report.html"
        report_path.write_text("<html><body>Test</body></html>")

        analysis_data: AnalysisData = {
            "config_name": "test",
            "config_key": "test",
            "stories_count": 5,
            "from_date": datetime(2025, 1, 1),
            "to_date": datetime(2025, 1, 2),
        }

        # Should return early without errors
        with patch("smtplib.SMTP") as smtp_mock:
            execute(report_path, analysis_data)
            smtp_mock.assert_not_called()

    def test_execute_enabled_variations(self, tmp_path, monkeypatch):
        """Test that EMAIL_ENABLED accepts multiple true values."""
        from after_analysis.email_sender import execute
        from newsbot.models import AnalysisData

        # Set up basic email configuration
        monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.test.com")
        monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
        monkeypatch.setenv("EMAIL_SENDER", "from@test.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "pwd")

        # Create email_reports directory and report
        email_reports_dir = tmp_path / "email_reports"
        email_reports_dir.mkdir()
        report_path = tmp_path / "test_report.html"
        email_report_path = email_reports_dir / "test_report.html"
        email_report_path.write_text("<html><body>Test</body></html>")

        analysis_data: AnalysisData = {
            "config_name": "test",
            "config_key": "test",
            "stories_count": 5,
            "from_date": datetime(2025, 1, 1),
            "to_date": datetime(2025, 1, 2),
        }

        for enabled_value in ["true", "True", "TRUE", "1", "yes", "Yes"]:
            monkeypatch.setenv("EMAIL_ENABLED", enabled_value)

            # Mock the database call to return recipients
            with (
                patch(
                    "after_analysis.email_sender.get_recipients_for_config",
                    return_value=["test@example.com"],
                ),
                patch(
                    "after_analysis.email_sender.get_available_newsletters",
                    return_value=[],
                ),
                patch(
                    "after_analysis.email_sender.CSSInliner",
                ) as inliner_mock,
                patch(
                    "after_analysis.email_sender._send_via_emailjs",
                ) as emailjs_mock,
                patch("smtplib.SMTP") as smtp_mock,
            ):
                inliner_mock.return_value.inline.side_effect = lambda html: html
                smtp_instance = Mock()
                smtp_instance.__enter__ = lambda self: smtp_instance
                smtp_instance.__exit__ = Mock(return_value=False)
                smtp_mock.return_value = smtp_instance

                execute(report_path, analysis_data)

                # Should attempt to send email (via SMTP or EmailJS depending on EMAIL_PROVIDER)
                assert smtp_mock.called or emailjs_mock.called, (
                    "Expected either SMTP or EmailJS send to be called"
                )
                smtp_mock.reset_mock()
                emailjs_mock.reset_mock()
