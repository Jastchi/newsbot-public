"""Test script to run the email sender hook independently."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from newsbot.constants import TZ

if TYPE_CHECKING:
    from newsbot.models import AnalysisData

sys.path.insert(0, str(Path(__file__).parent))
from after_analysis import email_sender

# Set up logging to see what happens
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test_email_hook() -> None:
    """Test the email sender hook with a sample report."""
    # Find an existing report file (or create a test one)
    reports_dir = Path("reports") / "Technology"

    # Look for the most recent report
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.html"))
        if report_files:
            # Use the most recent report
            report_path = max(report_files, key=lambda p: p.stat().st_mtime)
            logging.debug(f"Using existing report: {report_path}")
        else:
            logging.error("No report files found in the reports directory.")
            sys.exit(1)
    else:
        logging.error(
            "No reports directory found. Please run an analysis first.",
        )
        sys.exit(1)

    # Create sample analysis data
    analysis_data: AnalysisData = cast(
        "AnalysisData",
        {
            "success": True,
            "timestamp": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": "Technology",
            "articles_count": 42,
            "stories_count": 5,
            "duration": 12.34,
            "format": "html",
            "from_date": datetime.now(TZ),
            "to_date": datetime.now(TZ),
        },
    )

    # Run the email hook
    logging.debug("\nRunning email hook...")
    logging.debug(f"Report path: {report_path}")
    logging.debug(f"Analysis data: {analysis_data}")
    logging.debug(
        "\nMake sure you have configured EMAIL_* environment variables!",
    )

    email_sender.execute(report_path, analysis_data)
    logging.debug("\nDone! Check your email and the logs above.")


if __name__ == "__main__":
    test_email_hook()
