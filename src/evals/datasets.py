"""Typed, validated loaders for the labelled datasets in data/."""

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from evals.results import EvalError
from newsbot.models import Article

DATA_DIR = Path(__file__).parent / "data"
BASELINES_DIR = Path(__file__).parent / "baselines"
MIN_ARTICLES_PER_STORY = 2
# Articles need dates; clustering here does not look at them
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class _Strict(BaseModel):
    """Base for the dataset models: a misspelt key is an error."""

    model_config = ConfigDict(extra="forbid")


class LabelledArticle(_Strict):
    """An article and the story a human put it in."""

    id: str
    story: str | None
    source: str
    title: str
    content: str

    def to_article(self) -> Article:
        """Return the pipeline's Article for this row."""
        return Article(
            title=self.title,
            content=self.content,
            source=self.source,
            url=f"https://eval.invalid/{self.id}",
            published_date=_EPOCH,
            scraped_date=_EPOCH,
        )


class ClusteringDataset(_Strict):
    """Articles labelled with their story (None = no story)."""

    description: str
    # what the articles are written in; a config in another language
    # cannot be scored on them
    language: str = "en"
    articles: list[LabelledArticle]

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = Counter(a.id for a in self.articles)
        if duplicated := sorted(i for i, n in ids.items() if n > 1):
            msg = f"duplicate article ids: {duplicated}"
            raise ValueError(msg)
        sizes = Counter(a.story for a in self.articles if a.story)
        if lonely := sorted(
            s for s, n in sizes.items() if n < MIN_ARTICLES_PER_STORY
        ):
            msg = f"stories with a single article: {lonely}"
            raise ValueError(msg)
        return self

    def stories(self) -> dict[str, list[Article]]:
        """Group the articles by story, skipping unlabelled ones."""
        grouped: dict[str, list[Article]] = {}
        for row in self.articles:
            if row.story:
                grouped.setdefault(row.story, []).append(row.to_article())
        return grouped


class FaithfulnessCase(_Strict):
    """A candidate summary of one story, labelled by a human."""

    id: str
    story: str
    label: Literal["faithful", "unfaithful"]
    error: str | None
    summary: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.label == "unfaithful") != bool(self.error):
            msg = f"{self.id}: error is set exactly when unfaithful"
            raise ValueError(msg)
        return self

    @property
    def is_unfaithful(self) -> bool:
        """Whether the human label says the summary is wrong."""
        return self.label == "unfaithful"


class FaithfulnessDataset(_Strict):
    """Summaries whose source text is a story in the clustering data."""

    description: str
    cases: list[FaithfulnessCase]


def _read[T: BaseModel](model: type[T], data_dir: Path, name: str) -> T:
    try:
        return model.model_validate_json(
            (data_dir / name).read_text(encoding="utf-8"),
        )
    except OSError as err:
        msg = f"cannot read {name} from {data_dir}: {err}"
        raise EvalError(msg) from err
    except ValidationError as err:
        msg = f"{name} is invalid:\n{err}"
        raise EvalError(msg) from err


def load_clustering(data_dir: Path = DATA_DIR) -> ClusteringDataset:
    """Load and validate clustering.json from a data directory."""
    return _read(ClusteringDataset, data_dir, "clustering.json")


def load_faithfulness(
    clustering: ClusteringDataset,
    data_dir: Path = DATA_DIR,
) -> FaithfulnessDataset:
    """
    Load and validate faithfulness.json from a data directory.

    Args:
        clustering: The clustering data, whose stories supply each
            case's source articles.
        data_dir: Directory holding the file.

    Returns:
        The dataset, with every case tied to a known story.

    Raises:
        EvalError: If the file is invalid or a case names a story that
            is not in the clustering data.

    """
    dataset = _read(FaithfulnessDataset, data_dir, "faithfulness.json")
    known = {a.story for a in clustering.articles if a.story}
    if unknown := sorted({c.story for c in dataset.cases} - known):
        msg = f"faithfulness.json uses unknown stories: {unknown}"
        raise EvalError(msg)
    return dataset
