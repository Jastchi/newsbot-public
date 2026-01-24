"""
Django setup module for external scripts.

This module must be imported before any Django model imports.
It ensures Django is properly configured and initialized.
"""

import os
import sys
from pathlib import Path

# This file is at: src/utilities/django_setup.py
# __file__.parent.parent gets us to src/
_src_path = Path(__file__).resolve().parent.parent

# Add src to path FIRST so we can import web.newsserver.models
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Now setup Django - use web.web.settings since web is now the package
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.web.settings")

# Import and setup Django
# Note: django.setup() is idempotent, safe to call multiple times
# This is a Django setup module, so importing django is required
import django
import django.apps

# Setup Django (idempotent - safe to call multiple times)
# Only call setup if Django apps are not already ready
if not django.apps.apps.ready:
    django.setup()

DJANGO_SETUP_COMPLETE: bool = True
