"""Score summary-faithfulness detectors against human labels."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from evals.configs import Subject
from evals.datasets import FaithfulnessCase
from evals.metrics import DetectionScores, ungrounded_facts
from evals.offline import without_llm_provider
from evals.results import EvalError, Result
from newsbot.agents.name_validation_agent import NameValidationAgent
from newsbot.agents.summarization_agent import SummarizationAgent
from newsbot.model_cache import get_spacy_model
from newsbot.models import Article
from utilities import models as config_models

# What the pipeline tells the judge a story summary should be
STORY_SUMMARY_CONTEXT = (
    "A cumulative story summary. ONLY the summary text, "
    "no preambles or meta-commentary."
)

# Detector (and metric prefix) of the billed LLM judge
JUDGE = "judge"

# Flags a case as unfaithful, given the articles it summarises
Detector = Callable[[FaithfulnessCase, list[Article]], bool]


@dataclass(frozen=True)
class JudgeOptions:
    """The billed LLM judge to run, if any."""

    provider: str
    model: str


def run(
    judge: JudgeOptions | None = None,
    subject: Subject | None = None,
) -> Result:
    """
    Score every detector on the labelled summaries.

    Args:
        judge: Also run the LLM judge. This makes one API call per
            case, which may be billed; leave it out for the free,
            offline checks.
        subject: The config and data to run on; default is the
            production defaults and the seed data.

    Returns:
        Metrics per detector, with confusion counts and recall per
        injected error type.

    Raises:
        EvalError: If the judge failed open on any call, since its
            verdicts then say nothing about its quality; if the
            config's spaCy model is missing; or if the data is in
            another language than the config.

    """
    subject = subject or Subject()
    clustering, dataset = subject.load_faithfulness()
    cases = dataset.cases
    stories = clustering.stories()
    config = _config(judge, subject.config)
    # The pipeline's own source builders, so every detector sees the
    # same text. They do not use self, so the free run skips building
    # the whole agent; the judge run needs it for its judge. The name
    # check is the pipeline's code too, but only its matching is used,
    # so it gets no real LLM provider.
    summarizer = (
        SummarizationAgent(config)
        if judge
        else object.__new__(SummarizationAgent)
    )
    with without_llm_provider():
        names = NameValidationAgent(config)
    try:
        # fail on a missing model now, with the name the agent resolved
        get_spacy_model(names.spacy_model_name)
    except RuntimeError as err:
        raise EvalError(str(err)) from err

    def numbers_dates(case: FaithfulnessCase, articles: list[Article]) -> bool:
        source = summarizer._build_judge_source(articles)
        return bool(ungrounded_facts(case.summary, source))

    def unknown_names(case: FaithfulnessCase, articles: list[Article]) -> bool:
        sources = summarizer._get_article_contents(articles)
        return not names.validate_entities(case.summary, sources).is_valid

    detectors: dict[str, Detector] = {
        "numbers_dates": numbers_dates,
        "names": unknown_names,
    }
    detail: dict = subject.detail()
    if subject.key:
        # what a config can change; the default run has no such record
        detail["config"] = {
            "spacy_model": names.spacy_model_name,
            **(
                {"judge_provider": judge.provider, "judge_model": judge.model}
                if judge
                else {}
            ),
        }
    if judge:
        detectors[JUDGE] = _judge_detector(summarizer, detail)

    flags = {
        name: [detector(case, stories[case.story]) for case in cases]
        for name, detector in detectors.items()
    }
    flags["deterministic"] = [
        dates or unknown
        for dates, unknown in zip(
            flags["numbers_dates"],
            flags["names"],
            strict=True,
        )
    ]

    if judge:
        detail["judge_counters"] = summarizer.judge_agent.stats.as_dict()

    # the free run has no judge metrics for a baseline to compare
    return _score(cases, flags, detail, () if judge else (f"{JUDGE}.",))


def _config(
    judge: JudgeOptions | None,
    base: config_models.ConfigModel,
) -> config_models.ConfigModel:
    llm = base.llm.model_copy(
        update={
            # the offline checks must not need an API key
            "provider": judge.provider if judge else "ollama",
            "judge_model": judge.model if judge else "",
            "judge_enabled": judge is not None,
            "judge_check_source": True,
            "name_validation_enabled": True,
        },
    )
    return base.model_copy(update={"llm": llm})


def _score(
    cases: list[FaithfulnessCase],
    flags: dict[str, list[bool]],
    detail: dict,
    unscored: tuple[str, ...],
) -> Result:
    actual = [case.is_unfaithful for case in cases]
    metrics: dict[str, float] = {}
    for name, flagged in flags.items():
        scores = DetectionScores.from_flags(actual, flagged)
        metrics |= {f"{name}.{k}": v for k, v in scores.as_metrics().items()}
        detail[f"{name}_counts"] = scores.counts()
        detail[f"{name}_recall_by_error"] = _recall_by_error(cases, flagged)
    return Result("faithfulness", metrics, detail, unscored)


def _judge_detector(summarizer: SummarizationAgent, detail: dict) -> Detector:
    judge = summarizer.judge_agent
    detail["judge_model"] = judge.judge_model

    def flagged_by_judge(
        case: FaithfulnessCase,
        articles: list[Article],
    ) -> bool:
        verdict = judge.validate_output(
            case.summary,
            STORY_SUMMARY_CONTEXT,
            source=summarizer._build_judge_source(articles),
        )
        # Stop at the first failure instead of paying for the rest
        if judge.stats.judge_fail_open_rate():
            msg = (
                "the judge failed open, so it was not scored: "
                f"{judge.stats.as_dict()}"
            )
            raise EvalError(msg)
        return not verdict.is_valid

    return flagged_by_judge


def _recall_by_error(
    cases: list[FaithfulnessCase],
    flagged: list[bool],
) -> dict[str, float]:
    """Share of each injected error type the detector flagged."""
    by_error: dict[str, list[bool]] = defaultdict(list)
    for case, hit in zip(cases, flagged, strict=True):
        if case.error:
            by_error[case.error].append(hit)
    return {e: sum(v) / len(v) for e, v in sorted(by_error.items())}
