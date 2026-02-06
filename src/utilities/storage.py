"""Supabase storage utility for report uploads and downloads."""

import logging
import os
import time
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from supabase import Client, create_client

from utilities.django_models import NewsConfig

logger = logging.getLogger(__name__)

# Constants
EXPECTED_USERNAME_PARTS = 2


class SupabaseConfig(NamedTuple):
    """Supabase configuration extracted from DATABASE_URL."""

    project_ref: str
    service_key: str
    url: str


def parse_database_url(database_url: str | None) -> SupabaseConfig | None:
    """
    Parse DATABASE_URL to extract Supabase configuration.

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        SupabaseConfig if URL is Supabase, None otherwise

    Example:
        postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-region.pooler.supabase.com:5432/postgres

    """
    if not database_url or "supabase" not in database_url.lower():
        logger.info("DATABASE_URL does not point to Supabase")
        return None

    try:
        parsed = urlparse(database_url)

        # Check if hostname contains supabase
        if not parsed.hostname or "supabase" not in parsed.hostname:
            return None

        # Extract username (format: postgres.PROJECT_REF)
        if not parsed.username or "." not in parsed.username:
            logger.warning(
                "DATABASE_URL has unexpected username format, "
                "expected 'postgres.PROJECT_REF'",
            )
            return None

        # Split username to get project reference
        username_parts = parsed.username.split(".", 1)
        if len(username_parts) != EXPECTED_USERNAME_PARTS:
            logger.warning(
                "Could not extract project ref from username: "
                f"{parsed.username}",
            )
            return None

        project_ref = username_parts[1]
        service_key = parsed.password or ""
        url = f"https://{project_ref}.supabase.co/"

        return SupabaseConfig(
            project_ref=project_ref, service_key=service_key, url=url,
        )

    except Exception:
        logger.exception("Failed to parse DATABASE_URL")
        return None


def get_supabase_client() -> Client | None:
    """
    Create Supabase client from DATABASE_URL environment variable.

    Returns:
        Supabase client if DATABASE_URL points to Supabase,
        None otherwise

    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not service_key:
        logger.warning("SUPABASE_SERVICE_KEY is not set")
        return None

    config = parse_database_url(database_url)
    if not config:
        return None

    try:
        return create_client(config.url, service_key)
    except Exception:
        logger.exception("Failed to create Supabase client")
        return None


def should_use_supabase_for_config(config_name: str) -> bool:
    """
    Determine if a config should use Supabase storage.

    Queries the NewsConfig model in the database to check if
    the config's database_url points to Supabase.

    Args:
        config_name: Name/key of the configuration

    Returns:
        True if config has database_url pointing to Supabase,
        or if DATABASE_URL environment variable points to Supabase

    """
    # First check if DATABASE_URL points to Supabase
    database_url = os.getenv("DATABASE_URL", "")
    if "supabase" not in database_url.lower():
        return False

    # Try to look up NewsConfig in the database
    try:
        news_config = NewsConfig.objects.filter(key=config_name).first()

        if news_config is None:
            # No config found in database, default to Supabase
            logger.info(
                f"No NewsConfig found for '{config_name}', "
                f"defaulting to Supabase storage",
            )
            return True

        # Check if the config's database_url points to Supabase
        db_url = news_config.database_url

        # Handle environment variable expansion
        if db_url.startswith("${") and db_url.endswith("}"):
            env_var = db_url[2:-1]
            db_url = os.getenv(env_var, "")

        return "supabase" in db_url.lower()

    except Exception:
        logger.exception(f"Error looking up NewsConfig for '{config_name}'")
        # Default to Supabase on error (same as before)
        return True


def upload_to_supabase(
    client: Client,
    bucket: str,
    file_path: Path,
    destination_path: str,
    *,
    retry: bool = True,
) -> bool:
    """
    Upload a file to Supabase storage bucket.

    Args:
        client: Supabase client
        bucket: Bucket name
        file_path: Local file path to upload
        destination_path: Destination path in bucket
        retry: Whether to retry once on failure after cooldown

    Returns:
        True if upload succeeded, False otherwise

    """
    try:
        with file_path.open("rb") as f:
            content = f.read()

        client.storage.from_(bucket).upload(
            path=destination_path,
            file=content,
            file_options={"content-type": "text/html"},
        )
        logger.info(
            f"Uploaded {file_path.name} to {bucket}/{destination_path}",
        )
    except Exception as e:
        logger.warning(
            f"Failed to upload {file_path.name} to Supabase: {e}",
        )

        if retry:
            logger.info("Retrying upload after 3 second cooldown...")
            time.sleep(3)
            return upload_to_supabase(
                client, bucket, file_path, destination_path, retry=False,
            )

        logger.exception(
            f"Upload failed after retry for {file_path.name}",
        )
        return False
    else:
        return True


def list_supabase_reports(
    client: Client, bucket: str, config_name: str,
) -> list[dict]:
    """
    List all reports for a config in Supabase storage.

    Args:
        client: Supabase client
        bucket: Bucket name
        config_name: Configuration name (used as prefix)

    Returns:
        List of file metadata with 'name' and 'updated_at'

    """
    try:
        # List files in the config's directory
        files = client.storage.from_(bucket).list(path=config_name)

        # Filter for HTML files and exclude email_reports subdirectory
        return [
            f
            for f in files
            if f["name"].endswith(".html")
            and "email_reports" not in f["name"]
        ]
    except Exception:
        logger.exception(
            f"Failed to list reports for '{config_name}' from Supabase",
        )
        return []


def get_signed_url(
    client: Client, bucket: str, file_path: str, expires_in: int = 3600,
) -> str | None:
    """
    Generate a signed URL for a private bucket file.

    Args:
        client: Supabase client
        bucket: Bucket name
        file_path: File path in bucket
        expires_in: URL expiration time in seconds (default 1 hour)

    Returns:
        Signed URL string or None on failure

    """
    try:
        response = client.storage.from_(bucket).create_signed_url(
            path=file_path, expires_in=expires_in,
        )
    except Exception:
        logger.exception(f"Failed to generate signed URL for {file_path}")
        return None
    else:
        return response.get("signedURL")


def download_from_supabase(
    client: Client, bucket: str, file_path: str,
) -> bytes | None:
    """
    Download file content from Supabase storage.

    Args:
        client: Supabase client
        bucket: Bucket name
        file_path: File path in bucket

    Returns:
        File content as bytes or None on failure

    """
    try:
        return client.storage.from_(bucket).download(path=file_path)
    except Exception:
        logger.exception(f"Failed to download {file_path} from Supabase")
        return None
