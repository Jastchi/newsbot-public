"""
Supabase Upload Hook - Upload analysis reports to Supabase storage.

This hook automatically uploads the generated HTML report and its
corresponding email report to the specified Supabase bucket.
"""

import logging
from pathlib import Path

from newsbot.models import AnalysisData
from utilities.storage import get_supabase_client, upload_to_supabase

logger = logging.getLogger(__name__)


def execute(report_path: Path, analysis_data: AnalysisData) -> None:
    """
    Upload the analysis report and email report to Supabase.

    Args:
        report_path: Path to the generated report file
        analysis_data: Dictionary containing analysis metadata

    """
    client = get_supabase_client()
    if not client:
        logger.debug("Supabase client not available, skipping upload hook")
        return

    config_name = analysis_data.get("config_name", "default")
    bucket = "Reports"
    filename = report_path.name

    try:
        # 1. Upload the main report
        main_dest_path = f"{config_name}/{filename}"
        logger.info(f"Uploading main report to Supabase: {main_dest_path}")

        success = upload_to_supabase(
            client=client,
            bucket=bucket,
            file_path=report_path,
            destination_path=main_dest_path,
        )

        if not success:
            logger.warning(
                f"Failed to upload main report {filename} to Supabase",
            )

        # 2. Check for and upload the email report
        # Email reports are expected in the 'email_reports' subdirectory
        email_report_path = report_path.parent / "email_reports" / filename

        if email_report_path.exists():
            email_dest_path = f"{config_name}/email_reports/{filename}"
            logger.info(
                f"Uploading email report to Supabase: {email_dest_path}",
            )

            success_email = upload_to_supabase(
                client=client,
                bucket=bucket,
                file_path=email_report_path,
                destination_path=email_dest_path,
            )

            if not success_email:
                logger.warning(
                    f"Failed to upload email report {filename} to Supabase",
                )
        else:
            logger.debug(
                f"No email report found at {email_report_path}, "
                "skipping email upload",
            )

    except Exception:
        logger.exception(f"Error in Supabase upload hook for {filename}")
