"""Test script to run the Supabase upload hook independently."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from newsbot.constants import TZ

if TYPE_CHECKING:
    from newsbot.models import AnalysisData

# Add the current directory to sys.path to resolve imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from after_analysis import supabase_upload

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def test_supabase_upload_hook() -> None:
    """Test the Supabase upload hook with a sample report."""
    # Find an existing report file (or create a dummy one)
    # We'll look for reports in the Test folder as an example
    reports_dir = Path("reports") / "Test"

    if not reports_dir.exists():
        logging.warning(
            f"Reports directory {reports_dir} not found. "
            "Creating a dummy report for testing.",
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        dummy_report = reports_dir / "test_report.html"
        dummy_report.write_text(
            "<h1>Test Report</h1>"
            "<p>This is a test report for Supabase upload.</p>",
            encoding="utf-8",
        )
        report_path = dummy_report

        # Also create a dummy email report
        email_dir = reports_dir / "email_reports"
        email_dir.mkdir(parents=True, exist_ok=True)
        dummy_email = email_dir / "test_report.html"
        dummy_email.write_text(
            "<h1>Test Email Report</h1><p>This is a test email report.</p>",
            encoding="utf-8",
        )
    else:
        report_files = list(reports_dir.glob("*.html"))
        if report_files:
            report_path = max(report_files, key=lambda p: p.stat().st_mtime)
            logging.debug(f"Using existing report: {report_path}")
        else:
            logging.error("No report files found in the reports directory.")
            sys.exit(1)

    # Create sample analysis data
    analysis_data: AnalysisData = cast(
        "AnalysisData",
        {
            "success": True,
            "timestamp": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": "Test",
            "articles_count": 42,
            "stories_count": 5,
            "duration": 12.34,
            "format": "html",
            "from_date": datetime.now(TZ),
            "to_date": datetime.now(TZ),
        },
    )

    logging.info(f"Running Supabase upload hook for {report_path}...")
    supabase_upload.execute(report_path, analysis_data)
    logging.info("Hook execution complete.")

if __name__ == "__main__":
    test_supabase_upload_hook()
