r"""
Create a staff/superuser Subscriber that can log in only via Google.

No password is set; the subscriber must sign in with Google (email must
match). Use this to grant admin access: after running, sign in with
Google using that email to access the Django admin.

Usage (from project root):
  uv run src/web/manage.py create_staff_subscriber \\
      --email admin@example.com

  uv run src/web/manage.py create_staff_subscriber \\
      --email admin@example.com --first-name Admin --last-name User
"""

import argparse

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

Subscriber = get_user_model()


class Command(BaseCommand):
    """Create a staff/superuser Subscriber (Google-only login)."""

    help = (
        "Create a Subscriber with is_staff and is_superuser; no password. "
        "They must log in via Google (email must match) to access the admin."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add --email, --first-name, --last-name, --force arguments."""
        parser.add_argument(
            "--email",
            required=True,
            help=(
                "Email for the staff subscriber (required; "
                "use same email in Google)"
            ),
        )
        parser.add_argument(
            "--first-name",
            default="",
            help="First name (optional)",
        )
        parser.add_argument(
            "--last-name",
            default="",
            help="Last name (optional)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Update existing subscriber to staff/superuser if they exist",
        )

    def handle(self, *_args: object, **options: object) -> None:
        """Create or update the staff subscriber."""
        email = str(options.get("email") or "").strip().lower()
        first_name = str(options.get("first_name") or "").strip()
        last_name = str(options.get("last_name") or "").strip()
        force = bool(options.get("force", False))

        if not email:
            self.stdout.write(self.style.ERROR("Email is required."))
            return

        existing = Subscriber.objects.filter(email__iexact=email).first()
        if existing:
            if force:
                existing.first_name = first_name
                existing.last_name = last_name
                existing.is_staff = True
                existing.is_superuser = True
                existing.set_unusable_password()
                existing.save()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Updated {email} to staff/superuser (Google-only).",
                    ),
                )
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f"Subscriber with email {email} already exists. "
                        "Use --force to make them staff/superuser.",
                    ),
                )
            return

        subscriber = Subscriber.objects.create_user(
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=None,
            is_staff=True,
            is_superuser=True,
        )
        subscriber.set_unusable_password()
        subscriber.save()
        self.stdout.write(
            self.style.SUCCESS(
                f"Created staff subscriber: {email}. "
                "Sign in with Google using this email to access the admin.",
            ),
        )
