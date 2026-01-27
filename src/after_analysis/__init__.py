"""
Post-Analysis Hook System for NewsBot.

This module provides a plugin system for running custom actions after
the weekly analysis is complete. Hooks are automatically discovered and
executed for each generated report.

How to Create a Hook:
--------------------
1. Create a new Python file in this directory (e.g., my_hook.py)
2. Define an `execute(report_path, analysis_data)` function or create a
   class with an `execute` method and instantiate it as `hook`
3. The hook will be automatically discovered and run after each analysis

Example Hook:
------------
```python
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def execute(report_path: Path, analysis_data: dict) -> None:
    '''Do something with the generated report'''
    logger.info(f"Processing report: {report_path}")
    # Your custom logic here
```

Disabling Hooks:
---------------
To temporarily disable a hook, prefix the filename with underscore
(e.g., _my_hook.py)
"""

import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from newsbot.constants import TZ
from newsbot.models import AnalysisData

logger = logging.getLogger(__name__)


@runtime_checkable
class HookExecutable(Protocol):
    """Protocol for hook execute methods."""

    def __call__(
        self,
        report_path: Path,
        analysis_data: AnalysisData,
    ) -> None: ...


def run_hooks(
    report_path: Path,
    analysis_data: AnalysisData | None = None,
) -> None:
    """
    Discover and run all registered post-analysis hooks.

    Args:
        report_path: Path to the generated report file
        analysis_data: Optional metadata about the analysis run
            - success: bool, whether analysis succeeded
            - articles_count: int, number of articles analyzed
            - stories_count: int, number of stories identified
            - duration: float, analysis duration in seconds
            - timestamp: str, when the analysis was run
            - format: str, report format (html/txt/md)

    """
    if analysis_data is None:
        # Create empty AnalysisData dict with required fields
        analysis_data: AnalysisData = {
            "success": False,
            "articles_count": 0,
            "stories_count": 0,
            "duration": 0.0,
            "timestamp": "",
            "format": "",
            "config_name": "",
            "from_date": datetime.now(TZ),
            "to_date": datetime.now(TZ),
            "email_receivers_override": None,
        }

    hooks_dir = Path(__file__).parent
    hooks_executed = 0
    hooks_failed = 0

    logger.info(f"Running post-analysis hooks for report: {report_path.name}")

    # Discover all Python files in this directory (except __init__.py
    # and files starting with _)
    for hook_file in sorted(hooks_dir.glob("*.py")):
        if hook_file.name.startswith("_"):
            continue

        try:
            # Import the hook module
            module_name = f"after_analysis.{hook_file.stem}"
            module = importlib.import_module(module_name)

            # Look for a hook object with execute method or a standalone
            # execute function
            hook_callable: HookExecutable | None = None

            # Look for a hook object with execute method or a standalone
            # execute function
            if (
                hasattr(module, "hook")
                and hasattr(module.hook, "execute")
                and isinstance(module.hook.execute, HookExecutable)
            ):
                logger.info(f"Running hook: {hook_file.stem} (class-based)")
                hook_callable = module.hook.execute
            elif hasattr(module, "execute") and isinstance(
                module.execute,
                HookExecutable,
            ):
                logger.info(f"Running hook: {hook_file.stem} (function-based)")
                hook_callable = module.execute
            else:
                hook_callable = None
                logger.warning(
                    f"Hook {hook_file.stem} has no 'execute' function or "
                    "'hook.execute' method. Skipping.",
                )

            # Execute the hook if found
            if hook_callable is not None:
                hook_callable(report_path, analysis_data)
                hooks_executed += 1

        except Exception:
            hooks_failed += 1
            logger.exception(f"Error running hook {hook_file.stem}")

    # Summary
    if hooks_executed > 0 or hooks_failed > 0:
        logger.info(
            "Hooks complete: "
            f"{hooks_executed} succeeded, {hooks_failed} failed",
        )
    else:
        logger.debug("No post-analysis hooks found")


__all__ = ["run_hooks"]
