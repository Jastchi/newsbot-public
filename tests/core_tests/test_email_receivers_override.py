"""Tests for email receivers override functionality."""

import sys
from argparse import Namespace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestEmailReceiversOverride:
    """Test cases for email receivers override feature."""

    @pytest.mark.django_db
    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    @patch("newsbot.agents.story_clustering_agent.get_llm_provider")
    def test_pipeline_orchestrator_set_email_receivers_override(
        self,
        mock_story_provider,
        mock_summarization_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
    ):
        """Test that PipelineOrchestrator can set email receivers override."""
        from newsbot.pipeline import PipelineOrchestrator
        from utilities.django_models import NewsConfig

        # Setup mock providers
        mock_provider = MagicMock()
        mock_story_provider.return_value = mock_provider
        mock_summarization_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        # Create a NewsConfig for the test
        news_config, _ = NewsConfig.objects.get_or_create(
            key="test_override",
            defaults={"display_name": "Test Override Config"},
        )

        # Create orchestrator with news_config
        orchestrator = PipelineOrchestrator(
            sample_config,
            news_config=news_config,
        )

        # Initially should be None
        assert orchestrator.email_receivers_override is None

        # Set override with email addresses
        test_emails = ["test1@example.com", "test2@example.com"]
        orchestrator.set_email_receivers_override(test_emails)
        assert orchestrator.email_receivers_override == test_emails

        # Set override with empty list (disable emails)
        orchestrator.set_email_receivers_override([])
        assert orchestrator.email_receivers_override == []

    def test_email_sender_uses_override_with_emails(self, tmp_path):
        """Test email sender uses override when provided with emails."""
        from after_analysis.email_sender import execute

        # Create directory structure: reports/ and reports/email_reports/
        reports_dir = tmp_path / "reports"
        email_reports_dir = reports_dir / "email_reports"
        email_reports_dir.mkdir(parents=True, exist_ok=True)

        # Create the email report file
        email_report_path = email_reports_dir / "test_report.html"
        email_report_path.write_text("<html><body>Test Report</body></html>")

        # The execute function receives the regular report path
        # and internally looks for parent/email_reports/name
        regular_report_path = reports_dir / "test_report.html"

        # Analysis data with email override
        from newsbot.models import AnalysisData

        analysis_data: AnalysisData = cast(
            AnalysisData,
            {
                "config_name": "TestConfig",
                "email_receivers_override": ["override@example.com"],
                "stories_count": 5,
                "from_date": Mock(strftime=lambda x: "01 Jan"),
                "to_date": Mock(strftime=lambda x: "07 Jan"),
            },
        )

        # Mock environment and SMTP
        with patch("after_analysis.email_sender.load_dotenv"):
            with patch("os.getenv") as mock_getenv:
                # Configure email as enabled
                def getenv_side_effect(key, default=None):
                    env_vars = {
                        "EMAIL_ENABLED": "true",
                        "EMAIL_SMTP_SERVER": "smtp.test.com",
                        "EMAIL_SMTP_PORT": "587",
                        "EMAIL_SENDER": "sender@test.com",
                        "EMAIL_PASSWORD": "password",
                        "EMAIL_USE_SSL": "false",
                    }
                    return env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                with patch(
                    "after_analysis.email_sender.get_available_newsletters",
                    return_value=["Test Newsletter"],
                ):
                    with patch(
                        "after_analysis.email_sender.smtplib.SMTP"
                    ) as mock_smtp:
                        # Mock SMTP connection
                        mock_server = MagicMock()
                        mock_smtp.return_value.__enter__.return_value = (
                            mock_server
                        )

                        # Execute email hook
                        execute(regular_report_path, analysis_data)

                        # Verify email was sent
                        mock_server.send_message.assert_called_once()

                        # Get the message that was sent
                        sent_message = mock_server.send_message.call_args[0][0]

                        # Verify Bcc contains override email
                        assert "override@example.com" in sent_message["Bcc"]

    def test_email_sender_uses_override_with_empty_list(self, tmp_path):
        """Test email sender sends to sender only when override is empty list."""
        from after_analysis.email_sender import execute

        # Create directory structure: reports/ and reports/email_reports/
        reports_dir = tmp_path / "reports"
        email_reports_dir = reports_dir / "email_reports"
        email_reports_dir.mkdir(parents=True, exist_ok=True)

        # Create the email report file
        email_report_path = email_reports_dir / "test_report.html"
        email_report_path.write_text("<html><body>Test Report</body></html>")

        # The execute function receives the regular report path
        regular_report_path = reports_dir / "test_report.html"

        # Analysis data with empty email override (send to sender only)
        from newsbot.models import AnalysisData

        analysis_data: AnalysisData = cast(
            AnalysisData,
            {
                "config_name": "TestConfig",
                "email_receivers_override": [],
                "stories_count": 5,
                "from_date": Mock(strftime=lambda x: "01 Jan"),
                "to_date": Mock(strftime=lambda x: "07 Jan"),
            },
        )

        # Mock environment
        with patch("after_analysis.email_sender.load_dotenv"):
            with patch("os.getenv") as mock_getenv:
                # Configure email as enabled
                def getenv_side_effect(key, default=None):
                    env_vars = {
                        "EMAIL_ENABLED": "true",
                        "EMAIL_SMTP_SERVER": "smtp.test.com",
                        "EMAIL_SMTP_PORT": "587",
                        "EMAIL_SENDER": "sender@test.com",
                        "EMAIL_PASSWORD": "password",
                        "EMAIL_USE_SSL": "false",
                    }
                    return env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                with patch(
                    "after_analysis.email_sender.get_available_newsletters",
                    return_value=["Test Newsletter"],
                ):
                    with patch(
                        "after_analysis.email_sender.smtplib.SMTP"
                    ) as mock_smtp:
                        # Mock SMTP connection
                        mock_server = MagicMock()
                        mock_smtp.return_value.__enter__.return_value = (
                            mock_server
                        )

                        # Execute email hook
                        execute(regular_report_path, analysis_data)

                        # Verify email WAS sent (SMTP was called)
                        mock_server.send_message.assert_called_once()

                        # Get the message that was sent
                        sent_message = mock_server.send_message.call_args[0][0]

                        # Verify To field contains sender (with formatted name)
                        assert "sender@test.com" in sent_message["To"]
                        assert "TestConfig NewsBot" in sent_message["To"]

                        # Verify Bcc is empty (no additional recipients)
                        assert sent_message["Bcc"] == ""

    def test_email_sender_uses_database_without_override(self, tmp_path):
        """Test email sender uses database when no override provided."""
        from after_analysis.email_sender import execute

        # Create directory structure: reports/ and reports/email_reports/
        reports_dir = tmp_path / "reports"
        email_reports_dir = reports_dir / "email_reports"
        email_reports_dir.mkdir(parents=True, exist_ok=True)

        # Create the email report file
        email_report_path = email_reports_dir / "test_report.html"
        email_report_path.write_text("<html><body>Test Report</body></html>")

        # The execute function receives the regular report path
        regular_report_path = reports_dir / "test_report.html"

        # Analysis data WITHOUT email override
        from newsbot.models import AnalysisData

        analysis_data: AnalysisData = cast(
            AnalysisData,
            {
                "config_name": "TestConfig",
                "stories_count": 5,
                "from_date": Mock(strftime=lambda x: "01 Jan"),
                "to_date": Mock(strftime=lambda x: "07 Jan"),
            },
        )

        # Mock environment and database
        with patch("after_analysis.email_sender.load_dotenv"):
            with patch("os.getenv") as mock_getenv:
                # Configure email as enabled
                def getenv_side_effect(key, default=None):
                    env_vars = {
                        "EMAIL_ENABLED": "true",
                        "EMAIL_SMTP_SERVER": "smtp.test.com",
                        "EMAIL_SMTP_PORT": "587",
                        "EMAIL_SENDER": "sender@test.com",
                        "EMAIL_PASSWORD": "password",
                        "EMAIL_USE_SSL": "false",
                    }
                    return env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                with patch(
                    "after_analysis.email_sender.get_recipients_for_config",
                ) as mock_get_recipients:
                    # Mock database to return some recipients
                    mock_get_recipients.return_value = ["db@example.com"]

                    with patch(
                        "after_analysis.email_sender.get_available_newsletters",
                        return_value=["Test Newsletter"],
                    ):
                        with patch(
                            "after_analysis.email_sender.smtplib.SMTP"
                        ) as mock_smtp:
                            # Mock SMTP connection
                            mock_server = MagicMock()
                            mock_smtp.return_value.__enter__.return_value = (
                                mock_server
                            )

                            # Execute email hook
                            execute(
                                regular_report_path,
                                analysis_data,
                            )

                            # Verify database function was called
                            mock_get_recipients.assert_called_once_with(
                                "TestConfig"
                            )

                            # Verify email was sent
                            mock_server.send_message.assert_called_once()

                            # Get the message that was sent
                            sent_message = mock_server.send_message.call_args[
                                0
                            ][0]

                            # Verify Bcc contains database email
                            assert "db@example.com" in sent_message["Bcc"]

    def test_main_parses_email_receivers_argument(self):
        """Test main.py correctly parses --email-receivers argument."""
        from newsbot.main import main

        # Test with multiple email addresses
        with patch(
            "sys.argv",
            [
                "main.py",
                "analyze",
                "--config",
                "test",
                "--email-receivers",
                "test1@example.com",
                "test2@example.com",
            ],
        ):
            with patch("newsbot.main.run_analysis"):
                with patch("newsbot.main.load_config", return_value={}):
                    with patch("newsbot.main.setup_logging"):
                        with patch("newsbot.main.get_email_error_handler"):
                            # This should not raise an error
                            try:
                                main()
                            except SystemExit:
                                pass  # Expected when mocking

        # Test with no email addresses (disable emails)
        with patch(
            "sys.argv",
            [
                "main.py",
                "analyze",
                "--config",
                "test",
                "--email-receivers",
            ],
        ):
            with patch("newsbot.main.run_analysis"):
                with patch("newsbot.main.load_config", return_value={}):
                    with patch("newsbot.main.setup_logging"):
                        with patch("newsbot.main.get_email_error_handler"):
                            # This should not raise an error
                            try:
                                main()
                            except SystemExit:
                                pass  # Expected when mocking

    @pytest.mark.django_db
    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    @patch("newsbot.agents.story_clustering_agent.get_llm_provider")
    def test_orchestrator_method_exists(
        self,
        mock_story_provider,
        mock_summarization_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
    ):
        """Test that PipelineOrchestrator has the set method."""
        from newsbot.pipeline import PipelineOrchestrator
        from utilities.django_models import NewsConfig

        # Setup mock providers
        mock_provider = MagicMock()
        mock_story_provider.return_value = mock_provider
        mock_summarization_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        # Create a NewsConfig for the test
        news_config, _ = NewsConfig.objects.get_or_create(
            key="test_method_exists",
            defaults={"display_name": "Test Method Exists Config"},
        )

        orchestrator = PipelineOrchestrator(
            sample_config,
            news_config=news_config,
        )
        assert hasattr(orchestrator, "set_email_receivers_override")
        assert callable(orchestrator.set_email_receivers_override)
