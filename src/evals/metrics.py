"""Pure scoring functions for the evaluations. No models, no I/O."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import comb

from sklearn.metrics import adjusted_rand_score

# A metric may drop this much against a baseline before it counts as a
# regression; the seed sets are small, so one case moves a score ~0.03.
DEFAULT_TOLERANCE = 0.05

# English number format ("3,000", "4.25"), optionally scaled
# ("1.5 million"); a European "1,5" is not read. "1.5m" is not scaled:
# "m" as often means metres.
_NUMBER = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:[ \t-]+(thousand|million|billion|trillion)\b)?",
    re.IGNORECASE,
)
_SCALES = {
    "thousand": 10**3,
    "million": 10**6,
    "billion": 10**9,
    "trillion": 10**12,
}
_WORD = re.compile(r"[a-z]+")
_CARDINALS = (
    "two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
)
# "second" is left out: it is as often a unit of time as an ordinal
_ORDINALS = "third fourth fifth sixth seventh eighth ninth tenth"
_NUMBER_WORDS = {
    word: str(value) for value, word in enumerate(_CARDINALS.split(), start=2)
} | {word: str(value) for value, word in enumerate(_ORDINALS.split(), start=3)}
# "may", "march" and "august" are left out: common non-date words
_DATE_WORDS = frozenset(
    {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "april",
        "june",
        "july",
        "september",
        "october",
        "november",
        "december",
    },
)


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


@dataclass(frozen=True)
class ClusteringScores:
    """
    A clustering scored against hand-labelled stories.

    Attributes:
        ari: Adjusted Rand index of the two labelings.
        pair_precision: Share of predicted same-story pairs that really
            are the same story.
        pair_recall: Share of real same-story pairs that were found.

    """

    ari: float
    pair_precision: float
    pair_recall: float

    @property
    def pair_f1(self) -> float:
        """Harmonic mean of pairwise precision and recall."""
        return _f1(self.pair_precision, self.pair_recall)

    def as_metrics(self) -> dict[str, float]:
        """Return every score, all of them higher-is-better."""
        return {
            "ari": self.ari,
            "pair_precision": self.pair_precision,
            "pair_recall": self.pair_recall,
            "pair_f1": self.pair_f1,
        }


def clustering_scores(
    truth: Sequence[str | None],
    clusters: Sequence[Sequence[int]],
) -> ClusteringScores:
    """
    Score a clustering against hand-labelled stories.

    Articles with no story (truth None) and articles the algorithm left
    unclustered (noise) each count as a singleton, the usual convention
    for a clustering that may leave points out.

    Args:
        truth: Story name per article, or None for no story.
        clusters: Article indices per predicted cluster.

    Returns:
        ARI plus pairwise precision and recall.

    Raises:
        ValueError: If a cluster holds an unknown article or an article
            is in more than one cluster.

    """
    indices = [i for cluster in clusters for i in cluster]
    if not all(0 <= i < len(truth) for i in indices):
        msg = "a cluster holds an article index outside the labels"
        raise ValueError(msg)
    if len(indices) != len(set(indices)):
        msg = "an article is in more than one cluster"
        raise ValueError(msg)
    true_labels = [
        name if name is not None else f"noise:{i}"
        for i, name in enumerate(truth)
    ]
    pred_labels = [f"noise:{i}" for i in range(len(truth))]
    for cluster_id, members in enumerate(clusters):
        for i in members:
            pred_labels[i] = f"cluster:{cluster_id}"

    # Count same-story pairs from group sizes instead of listing pairs
    def pairs(labels: Sequence[object]) -> int:
        return sum(comb(n, 2) for n in Counter(labels).values())

    both = pairs(list(zip(true_labels, pred_labels, strict=True)))
    predicted, real = pairs(pred_labels), pairs(true_labels)
    return ClusteringScores(
        ari=float(adjusted_rand_score(true_labels, pred_labels)),
        pair_precision=both / predicted if predicted else 0.0,
        pair_recall=both / real if real else 0.0,
    )


@dataclass(frozen=True)
class DetectionScores:
    """
    A detector (judge or deterministic check) scored against labels.

    The positive class is "this output is bad", so recall is the share
    of bad outputs caught and specificity the share of good outputs
    left alone.

    Attributes:
        tp: Bad outputs flagged.
        fp: Good outputs flagged.
        fn: Bad outputs missed.
        tn: Good outputs left alone.

    """

    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def from_flags(
        cls,
        actual: Sequence[bool],
        flagged: Sequence[bool],
    ) -> "DetectionScores":
        """
        Count a detector's hits and misses.

        Args:
            actual: True where the human label says the output is bad.
            flagged: True where the detector flagged the output.

        Returns:
            The confusion counts.

        """
        pairs = list(zip(actual, flagged, strict=True))
        return cls(
            tp=sum(a and f for a, f in pairs),
            fp=sum(f and not a for a, f in pairs),
            fn=sum(a and not f for a, f in pairs),
            tn=sum(not a and not f for a, f in pairs),
        )

    @property
    def total(self) -> int:
        """Number of outputs scored."""
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        """Share of flagged outputs that were really bad."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Share of bad outputs that were flagged."""
        bad = self.tp + self.fn
        return self.tp / bad if bad else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        return _f1(self.precision, self.recall)

    @property
    def specificity(self) -> float:
        """Share of good outputs that were left alone."""
        good = self.tn + self.fp
        return self.tn / good if good else 0.0

    @property
    def accuracy(self) -> float:
        """Share of outputs the detector got right."""
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def kappa(self) -> float:
        """
        Cohen's kappa: agreement with the labels beyond chance.

        1 is perfect, 0 is no better than chance, negative is worse.
        """
        if not self.total:
            return 0.0
        by_chance = (
            (self.tp + self.fp) * (self.tp + self.fn)
            + (self.fn + self.tn) * (self.fp + self.tn)
        ) / self.total**2
        if by_chance == 1:  # both only ever say one thing
            return float(self.accuracy == 1)
        return (self.accuracy - by_chance) / (1 - by_chance)

    def as_metrics(self) -> dict[str, float]:
        """Return every rate, all of them higher-is-better."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
        }

    def counts(self) -> dict[str, int]:
        """Return the raw confusion counts."""
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


def _figure(number: str, scale: str) -> str:
    """Write a number the same way however the text spells it."""
    # Decimal, not float: float formatting merges large figures
    value = Decimal(number.replace(",", "")) * _SCALES.get(scale.lower(), 1)
    return format(value.normalize(), "f")


def _facts(text: str) -> set[str]:
    """Numbers, spelled-out numbers/ordinals and weekday/month names."""
    facts = {_figure(n, scale) for n, scale in _NUMBER.findall(text)}
    for word in _WORD.findall(text.lower()):
        if word in _NUMBER_WORDS:
            facts.add(_NUMBER_WORDS[word])
        elif word in _DATE_WORDS:
            facts.add(word)
    return facts


def ungrounded_facts(summary: str, source: str) -> set[str]:
    """
    Find numbers and dates in a summary that the source never states.

    Cheap and deterministic: it catches changed figures, weekdays and
    months, but not a wrong entity or a made-up clause with no digits
    in it. That is what the LLM judge is for.

    Args:
        summary: Text to check.
        source: Source article text.

    Returns:
        Normalised facts present in the summary but not the source.

    """
    return _facts(summary) - _facts(source)


def regressions(
    current: dict[str, float],
    baseline: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
    """
    List metrics that dropped against a saved baseline.

    Every metric must be higher-is-better. A baseline metric missing
    from the current run counts as a regression, so a detector that
    silently stopped running cannot pass.

    Args:
        current: Metrics from this run.
        baseline: Metrics from the saved baseline.
        tolerance: Allowed absolute drop.

    Returns:
        One message per regressed metric; empty if none.

    """
    return [
        f"{name}: {expected:.3f} -> "
        + (f"{current[name]:.3f}" if name in current else "missing")
        for name, expected in baseline.items()
        if name not in current or current[name] < expected - tolerance
    ]
