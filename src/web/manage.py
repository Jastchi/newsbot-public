#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys
from pathlib import Path

# This file is at: src/web/manage.py
_src_path = Path(__file__).resolve().parent.parent
_script_dir = Path(__file__).resolve().parent

# Python automatically adds the script's directory (src/web/) to
# paths at index 0. This causes 'web' to resolve to src/web/web/
# (settings pkg) instead of src/web/ (our package). We need src/ to come
#  so 'web' resolves to src/web/ and 'web.newsserver' works correctly.

# Remove script directory temporarily if present
_script_dir_str = str(_script_dir)
_script_dir_was_in_path = _script_dir_str in sys.path
if _script_dir_was_in_path:
    sys.path.remove(_script_dir_str)

# Add src to path at position 0
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Re-add script directory AFTER src (so src takes precedence for 'web')
if _script_dir_was_in_path:
    sys.path.append(_script_dir_str)

try:
    from django.core.management import execute_from_command_line
except ImportError as exc:
    raise ImportError(
        "Couldn't import Django. Are you sure it's installed and "
        "available on your PYTHONPATH environment variable? Did you "
        "forget to activate a virtual environment?",
    ) from exc


def main() -> None:
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.web.settings")

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
