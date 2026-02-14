"""
Create a test user, not a subscriber (for "Request to be added" flow).

The user can log in at /accounts/login/ with email + password.
They will see the Schedule page (if allowed by your view) and the
"You're not a subscriber yet" / "Request to be added" panel.

Usage (from project root):
  uv run src/web/manage.py create_test_user --password YOUR_PASSWORD

  uv run src/web/manage.py create_test_user --email other@example.com
      --password otherpass
"""

import argparse

from django.core.management.base import BaseCommand

from web.newsserver.models import Subscriber


class Command(BaseCommand):
    """Create a test user that is not a Subscriber."""

    help = (
        "Create a test user (email + password) that is not a Subscriber, "
        "for testing the 'Request to be added' flow."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add --email, --password, --force arguments."""
        parser.add_argument(
            "--email",
            required=True,
            help="Email for the test user (required)",
        )
        parser.add_argument(
            "--password",
            required=True,
            help="Password for the test user (required)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Recreate user and reset password if they already exist",
        )

    def handle(self, *_args: object, **options: object) -> None:
        """Create or update the test user."""
        email = str(options.get("email") or "").strip().lower()
        password: str = str(options["password"])
        force = bool(options.get("force", False))

        if not email:
            self.stdout.write(self.style.ERROR("Email is required."))
            return

        if Subscriber.objects.filter(email__iexact=email).exists():
            self.stdout.write(
                self.style.WARNING(
                    f"A Subscriber with email {email} already exists. "
                    "Use a different email or remove that Subscriber to test "
                    "'not subscribed' flow.",
                ),
            )
            if not force:
                return
            # With --force we ensure User exists (no Subscriber removal)
            self.stdout.write("Continuing with --force (User only).")

        existing = Subscriber.objects.filter(email__iexact=email).first()
        if existing:
            if force:
                existing.set_password(password)
                existing.is_staff = False
                existing.is_superuser = False
                existing.save()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Updated existing user {email} "
                        "(password reset, not staff).",
                    ),
                )
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f"User with email {email} already exists. "
                        "Use --force to reset password and ensure non-staff.",
                    ),
                )
            self._print_credentials(email, password)
            return

        Subscriber.objects.create_user(
            username=email,
            email=email,
            password=password,
            is_staff=False,
            is_superuser=False,
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Created test user: {email} (not a subscriber).",
            ),
        )
        self._print_credentials(email, password)

    def _print_credentials(self, email: str, password: str) -> None:
        self.stdout.write("")
        self.stdout.write("Log in at /accounts/login/ with:")
        self.stdout.write(f"  Email:    {email}")
        self.stdout.write(f"  Password: {password}")
        self.stdout.write("")
