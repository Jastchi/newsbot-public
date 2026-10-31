"""A run's results: saving, printing and comparing to a baseline."""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.metrics import regressions


class EvalError(Exception):
    """A run that cannot be scored; reported without a traceback."""


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class Result:
    """
    What an evaluation produced.

    Attributes:
        name: Which evaluation ran.
        metrics: Flat name -> score, every one of them higher-is-better,
            so a baseline comparison needs no per-metric direction.
        detail: Context that is saved but never compared: settings,
            confusion counts, recall per error type.
        unscored: Prefixes of metrics this run cannot produce, such as
            the judge's in a free run. A baseline comparison leaves
            those baseline metrics out.

    """

    name: str
    metrics: dict[str, float]
    detail: dict[str, Any] = field(default_factory=dict)
    unscored: tuple[str, ...] = ()

    def to_json(self) -> str:
        """Serialise the result with run metadata."""
        return (
            json.dumps(
                {
                    "metrics": self.metrics,
                    "detail": self.detail,
                    "meta": {
                        "eval": self.name,
                        "git_sha": _git_sha(),
                        "run_at": datetime.now()
                        .astimezone()
                        .isoformat(timespec="seconds"),
                    },
                },
                indent=2,
            )
            + "\n"
        )

    def table(self) -> str:
        """Return the metrics as aligned text for a terminal."""
        width = max(map(len, self.metrics), default=0)
        return "\n".join(
            f"{name:<{width}}  {value:.3f}"
            for name, value in self.metrics.items()
        )


@dataclass(frozen=True)
class Comparison:
    """A result compared to a saved baseline."""

    regressed: list[str]
    warnings: list[str]


def compare(result: Result, baseline: Path, tolerance: float) -> Comparison:
    """
    Compare a result to a saved run.

    Args:
        result: This run.
        baseline: A file written by an earlier run.
        tolerance: Allowed absolute drop per metric.

    Returns:
        The regressed metrics, and a warning when the settings differ or
        baseline metrics were left out (see Result.unscored).

    Raises:
        EvalError: If the baseline cannot be read, or nothing in it is
            left to compare.

    """
    try:
        saved = json.loads(baseline.read_text(encoding="utf-8"))
        saved_metrics = saved["metrics"]
    except (OSError, ValueError, KeyError, TypeError) as err:
        msg = f"cannot read a baseline from {baseline}: {err!r}"
        raise EvalError(msg) from err
    warnings = []
    kept = {
        k: v
        for k, v in saved_metrics.items()
        if not k.startswith(result.unscored)
    }
    if left_out := len(saved_metrics) - len(kept):
        warnings.append(
            f"left out of the comparison: {left_out} baseline metric(s) "
            f"starting {', '.join(result.unscored)}, which this run "
            "cannot score",
        )
    saved_metrics = kept
    if not saved_metrics:
        msg = (
            f"nothing to compare in {baseline}: it has no metrics, "
            "or this run cannot score any of them"
        )
        raise EvalError(msg)

    def identity(detail: dict) -> tuple:
        return detail.get("config"), detail.get("config_key")

    old, new = identity(saved.get("detail", {})), identity(result.detail)
    if old != new:
        warnings.append(f"baseline settings {old} differ from {new}")
    return Comparison(
        regressed=regressions(result.metrics, saved_metrics, tolerance),
        warnings=warnings,
    )
