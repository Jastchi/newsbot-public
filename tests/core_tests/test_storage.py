"""Tests for Supabase storage utilities."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from utilities.storage import (
    download_from_supabase,
    get_signed_url,
    get_supabase_client,
    list_supabase_reports,
    parse_database_url,
    should_use_supabase_for_config,
    upload_to_supabase,
)


class TestParseDatabaseUrl:
    """Test cases for parse_database_url function."""

    def test_valid_supabase_url(self):
        """Test parsing a valid Supabase DATABASE_URL."""
        url = (
            "postgresql://postgres.myproject123:mypassword"
            "@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        )
        result = parse_database_url(url)

        assert result is not None
        assert result.project_ref == "myproject123"
        assert result.service_key == "mypassword"
        assert result.url == "https://myproject123.supabase.co/"

    def test_non_supabase_url(self):
        """Test that non-Supabase URLs return None."""
        url = "postgresql://user:pass@localhost:5432/mydb"
        result = parse_database_url(url)
        assert result is None

    def test_empty_url(self):
        """Test that empty URL returns None."""
        assert parse_database_url("") is None
        assert parse_database_url(None) is None

    def test_url_without_hostname(self):
        """Test URL without hostname."""
        url = "postgresql://postgres.myproject:pass@:5432/postgres"
        result = parse_database_url(url)
        assert result is None

    def test_url_with_invalid_username_format(self):
        """Test URL with username not in 'postgres.PROJECT_REF' format."""
        # Username without a dot
        url = "postgresql://postgres:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        result = parse_database_url(url)
        assert result is None

    def test_url_with_multiple_dots_in_username(self):
        """Test URL with too many parts in username."""
        # Username with multiple dots (should still work, takes everything after first dot)
        url = (
            "postgresql://postgres.my.project.ref:pass"
            "@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        )
        result = parse_database_url(url)

        assert result is not None
        assert result.project_ref == "my.project.ref"

    def test_url_without_password(self):
        """Test URL without password."""
        url = "postgresql://postgres.myproject@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        result = parse_database_url(url)

        assert result is not None
        assert result.project_ref == "myproject"
        assert result.service_key == ""

    def test_malformed_url_exception_handling(self):
        """Test that malformed URLs are handled gracefully."""
        # This should trigger an exception during parsing
        url = "not-a-valid-url-at-all"
        result = parse_database_url(url)
        assert result is None


class TestGetSupabaseClient:
    """Test cases for get_supabase_client function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_database_url(self):
        """Test when DATABASE_URL is not set."""
        result = get_supabase_client()
        assert result is None

    @patch.dict(
        os.environ,
        {"DATABASE_URL": "postgresql://localhost/db"},
        clear=True,
    )
    def test_no_service_key(self):
        """Test when SUPABASE_SERVICE_KEY is not set."""
        result = get_supabase_client()
        assert result is None

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
            "SUPABASE_SERVICE_KEY": "my-service-key",
        },
        clear=True,
    )
    @patch("utilities.storage.create_client")
    def test_successful_client_creation(self, mock_create_client):
        """Test successful Supabase client creation."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        result = get_supabase_client()

        assert result == mock_client
        mock_create_client.assert_called_once_with(
            "https://myproject.supabase.co/",
            "my-service-key",
        )

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
            "SUPABASE_SERVICE_KEY": "my-service-key",
        },
        clear=True,
    )
    @patch("utilities.storage.create_client")
    def test_client_creation_exception(self, mock_create_client):
        """Test handling of exceptions during client creation."""
        mock_create_client.side_effect = Exception("Connection failed")

        result = get_supabase_client()

        assert result is None


class TestShouldUseSupabaseForConfig:
    """Test cases for should_use_supabase_for_config function."""

    @patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://localhost/db"}, clear=True
    )
    def test_non_supabase_database_url(self):
        """Test when DATABASE_URL doesn't point to Supabase."""
        result = should_use_supabase_for_config("test_config")
        assert result is False

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        },
        clear=True,
    )
    def test_no_config_file_defaults_to_supabase(self, tmp_path, monkeypatch):
        """Test that missing config file defaults to Supabase."""
        # Change working directory to temp path where config doesn't exist
        monkeypatch.chdir(tmp_path)

        result = should_use_supabase_for_config("nonexistent_config")
        assert result is True

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        },
        clear=True,
    )
    @pytest.mark.django_db
    def test_config_without_database_section(self, tmp_path, monkeypatch):
        """Test config without database section (empty database_url)."""
        from utilities.django_models import NewsConfig

        # Create NewsConfig with empty database_url (simulating no database section)
        NewsConfig.objects.create(
            key="test_config",
            display_name="Test Config",
            database_url="",
        )

        result = should_use_supabase_for_config("test_config")
        assert result is False

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        },
        clear=True,
    )
    @pytest.mark.django_db
    def test_config_with_supabase_database(self):
        """Test config with Supabase database URL."""
        from utilities.django_models import NewsConfig

        # Create NewsConfig with Supabase database URL
        NewsConfig.objects.create(
            key="test_config",
            display_name="Test Config",
            database_url="postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
        )

        result = should_use_supabase_for_config("test_config")
        assert result is True

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
        },
        clear=True,
    )
    @pytest.mark.django_db
    def test_config_not_found_defaults_to_supabase(self):
        """Test that missing config defaults to Supabase when DATABASE_URL is Supabase."""
        # Don't create any NewsConfig - test the "not found" path
        result = should_use_supabase_for_config("nonexistent_config")
        assert result is True

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        },
        clear=True,
    )
    @pytest.mark.django_db
    def test_config_with_non_supabase_database(self, tmp_path, monkeypatch):
        """Test config with non-Supabase database URL."""
        from utilities.django_models import NewsConfig

        # Create NewsConfig with non-Supabase database URL
        NewsConfig.objects.create(
            key="test_config",
            display_name="Test Config",
            database_url="postgresql://localhost/db",
        )

        result = should_use_supabase_for_config("test_config")
        assert result is False

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://postgres.myproject:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        },
        clear=True,
    )
    @pytest.mark.django_db
    def test_database_exception_defaults_to_supabase(self, monkeypatch):
        """Test that database exceptions default to Supabase."""
        from utilities.django_models import NewsConfig

        # Mock NewsConfig.objects.filter to raise an exception
        def mock_filter(*args, **kwargs):
            raise Exception("Database error")

        monkeypatch.setattr(NewsConfig.objects, "filter", mock_filter)

        # Database exception should default to True (Supabase)
        result = should_use_supabase_for_config("test_config")
        assert result is True


class TestUploadToSupabase:
    """Test cases for upload_to_supabase function."""

    def test_successful_upload(self, tmp_path):
        """Test successful file upload."""
        # Create a test file
        test_file = tmp_path / "test_report.html"
        test_file.write_text("<html><body>Test Report</body></html>")

        # Mock Supabase client
        mock_client = Mock()
        mock_storage = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.upload.return_value = {
            "path": "Reports/test/test_report.html"
        }

        result = upload_to_supabase(
            mock_client,
            "Reports",
            test_file,
            "test/test_report.html",
        )

        assert result is True
        mock_client.storage.from_.assert_called_once_with("Reports")
        mock_bucket.upload.assert_called_once()

    def test_upload_failure_with_retry(self, tmp_path):
        """Test upload failure triggers retry."""
        test_file = tmp_path / "test_report.html"
        test_file.write_text("<html><body>Test Report</body></html>")

        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        # First call fails, second succeeds
        mock_bucket.upload.side_effect = [
            Exception("Network error"),
            {"path": "Reports/test/test_report.html"},
        ]

        with patch("utilities.storage.time.sleep"):
            result = upload_to_supabase(
                mock_client,
                "Reports",
                test_file,
                "test/test_report.html",
            )

        assert result is True
        assert mock_bucket.upload.call_count == 2

    def test_upload_failure_no_retry(self, tmp_path):
        """Test upload failure without retry."""
        test_file = tmp_path / "test_report.html"
        test_file.write_text("<html><body>Test Report</body></html>")

        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.upload.side_effect = Exception("Network error")

        result = upload_to_supabase(
            mock_client,
            "Reports",
            test_file,
            "test/test_report.html",
            retry=False,
        )

        assert result is False
        assert mock_bucket.upload.call_count == 1

    def test_upload_failure_after_retry(self, tmp_path):
        """Test upload failure persists after retry."""
        test_file = tmp_path / "test_report.html"
        test_file.write_text("<html><body>Test Report</body></html>")

        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        # Both calls fail
        mock_bucket.upload.side_effect = Exception("Network error")

        with patch("utilities.storage.time.sleep"):
            result = upload_to_supabase(
                mock_client,
                "Reports",
                test_file,
                "test/test_report.html",
            )

        assert result is False
        assert mock_bucket.upload.call_count == 2


class TestListSupabaseReports:
    """Test cases for list_supabase_reports function."""

    def test_list_reports_success(self):
        """Test successful listing of reports."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket

        mock_files = [
            {
                "name": "news_report_20251201_120000.html",
                "updated_at": "2025-12-01T12:00:00Z",
            },
            {
                "name": "news_report_20251202_120000.html",
                "updated_at": "2025-12-02T12:00:00Z",
            },
            {
                "name": "email_reports/email_20251201.html",
                "updated_at": "2025-12-01T13:00:00Z",
            },
            {"name": "readme.txt", "updated_at": "2025-12-01T10:00:00Z"},
        ]
        mock_bucket.list.return_value = mock_files

        result = list_supabase_reports(mock_client, "Reports", "Technology")

        assert len(result) == 2
        assert all(r["name"].endswith(".html") for r in result)
        assert all("email_reports" not in r["name"] for r in result)
        mock_client.storage.from_.assert_called_once_with("Reports")
        mock_bucket.list.assert_called_once_with(path="Technology")

    def test_list_reports_empty(self):
        """Test listing when no reports exist."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.list.return_value = []

        result = list_supabase_reports(mock_client, "Reports", "EmptyConfig")

        assert result == []

    def test_list_reports_exception(self):
        """Test handling of exceptions during listing."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.list.side_effect = Exception("API error")

        result = list_supabase_reports(mock_client, "Reports", "Technology")

        assert result == []


class TestGetSignedUrl:
    """Test cases for get_signed_url function."""

    def test_get_signed_url_success(self):
        """Test successful signed URL generation."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.create_signed_url.return_value = {
            "signedURL": "https://example.supabase.co/storage/v1/object/sign/Reports/test.html?token=abc123"
        }

        result = get_signed_url(mock_client, "Reports", "test/test.html")

        assert result is not None
        assert "https://" in result
        assert "token=" in result
        mock_bucket.create_signed_url.assert_called_once_with(
            path="test/test.html",
            expires_in=3600,
        )

    def test_get_signed_url_custom_expiry(self):
        """Test signed URL with custom expiration."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.create_signed_url.return_value = {
            "signedURL": "https://example.com/signed"
        }

        result = get_signed_url(
            mock_client, "Reports", "test/test.html", expires_in=7200
        )

        assert result is not None
        mock_bucket.create_signed_url.assert_called_once_with(
            path="test/test.html",
            expires_in=7200,
        )

    def test_get_signed_url_exception(self):
        """Test handling of exceptions during URL generation."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.create_signed_url.side_effect = Exception("API error")

        result = get_signed_url(mock_client, "Reports", "test/test.html")

        assert result is None


class TestDownloadFromSupabase:
    """Test cases for download_from_supabase function."""

    def test_download_success(self):
        """Test successful file download."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_content = b"<html><body>Test Report</body></html>"
        mock_bucket.download.return_value = mock_content

        result = download_from_supabase(
            mock_client, "Reports", "test/test.html"
        )

        assert result == mock_content
        mock_client.storage.from_.assert_called_once_with("Reports")
        mock_bucket.download.assert_called_once_with(path="test/test.html")

    def test_download_exception(self):
        """Test handling of exceptions during download."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.download.side_effect = Exception("File not found")

        result = download_from_supabase(
            mock_client, "Reports", "test/nonexistent.html"
        )

        assert result is None
