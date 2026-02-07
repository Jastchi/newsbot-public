"""
Daily management command: send one email to the admin with new requests.

Finds SubscriberRequest records not yet included in a daily email, sends
one digest to EMAIL_ADMIN_NOTIFICATION_TO (or ADMIN_EMAIL),
then marks those records as included.

Schedule via cron, e.g.:
  0 9 * * * cd /path/to/project &&
      python -m django send_subscriber_request_digest

Or run manually:
  python -m django send_subscriber_request_digest
"""

import argparse
import logging
from datetime import UTC

from django.conf import settings
from django.core.mail import send_mail
from django.core.management.base import BaseCommand
from django.utils import timezone

from web.newsserver.models import SubscriberRequest

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Send one daily digest of new subscriber requests to the admin."""

    help = (
        "Find SubscriberRequest records not yet included in a digest, "
        "send one email to the admin, and mark them as included."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add --dry-run argument."""
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "List requests that would be included without sending "
                "or marking."
            ),
        )

    def handle(self, *_args: object, **options: object) -> None:
        """Find pending requests, send digest or dry-run, then mark."""
        dry_run = options["dry_run"]
        pending = list(
            SubscriberRequest.objects.filter(
                included_in_daily_email_at__isnull=True,
            ).order_by("created_at"),
        )

        if not pending:
            self.stdout.write("No new subscriber requests to report.")
            return

        admin_to = getattr(settings, "EMAIL_ADMIN_NOTIFICATION_TO", "").strip()
        if not admin_to:
            self.stdout.write(
                self.style.WARNING(
                    "EMAIL_ADMIN_NOTIFICATION_TO not set; "
                    "skipping send. Configure in settings or env.",
                ),
            )
            return

        lines = [
            "New subscriber signup requests:",
            "",
        ]
        for req in pending:
            name = f"{req.first_name} {req.last_name}".strip() or "(no name)"
            created = req.created_at.astimezone(UTC).strftime(
                "%Y-%m-%d %H:%M UTC",
            )
            lines.append(f"  - {req.email}  ({name})  requested {created}")
        body = "\n".join(lines)

        subject = f"NewsBot: {len(pending)} new subscriber request(s)"

        if dry_run:
            self.stdout.write(f"Dry run: would send to {admin_to}")
            self.stdout.write(body)
            return

        try:
            send_mail(
                subject=subject,
                message=body,
                from_email=settings.DEFAULT_FROM_EMAIL or None,
                recipient_list=[admin_to],
                fail_silently=False,
            )
        except Exception as e:
            logger.exception("Failed to send subscriber request digest")
            self.stdout.write(self.style.ERROR(f"Send failed: {e}"))
            return

        now = timezone.now()
        for req in pending:
            req.included_in_daily_email_at = now
            req.save(update_fields=["included_in_daily_email_at"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Sent digest with {len(pending)} request(s) to {admin_to}.",
            ),
        )
