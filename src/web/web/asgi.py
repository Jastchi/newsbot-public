"""
ASGI config for web project.

It exposes the ASGI callable as a module-level variable named
``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
import sys
from pathlib import Path

# Add src to path so web.newsserver can be found
# This file is at: src/web/web/asgi.py
# So __file__.parent.parent.parent gets us to src/
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# E402: Import after path manipulation is intentional
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.web.settings")

application = get_asgi_application()
