"""Score story clustering against hand-labelled stories."""

from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from evals.configs import Subject
from evals.metrics import clustering_scores
from evals.offline import without_llm_provider
from evals.results import EvalError, Result
from newsbot.agents.story_clustering_agent import StoryClusteringAgent
from utilities import models as config_models

# Settings a run may override; the rest are the production defaults
OPTIONS = ("algorithm", "similarity_threshold", "embedding_model")


def run(
    overrides: dict[str, Any],
    subject: Subject | None = None,
) -> Result:
    """
    Cluster the labelled articles with the production agent.

    Args:
        overrides: Clustering settings to change from the subject's,
            keyed by name (see OPTIONS).
        subject: The config and data to run on; default is the
            production defaults and the seed data.

    Returns:
        ARI and pairwise scores, with the settings used.

    Raises:
        EvalError: If the embedding model cannot be loaded, or the
            data is in another language than the config.

    """
    subject = subject or Subject()
    dataset = subject.load_clustering()
    articles = [row.to_article() for row in dataset.articles]
    settings = config_models.StoryClusteringConfigModel(
        **subject.config.story_clustering.model_dump() | overrides,
    )
    config = subject.config.model_copy(update={"story_clustering": settings})
    with without_llm_provider():
        agent = StoryClusteringAgent(config)
    if agent.embedding_model is None:
        msg = f"cannot load {config.story_clustering.embedding_model}"
        raise EvalError(msg)

    # The steps identify_top_stories runs before it samples clusters and
    # asks the LLM for titles. Tests fail if these methods go away.
    embeddings = agent.embedding_model.encode(
        agent._get_embedding_texts(articles),
    )
    similarity = agent._apply_geo_penalty_to_similarity(
        cosine_similarity(embeddings),
        agent._get_article_locations(articles),
    )
    clusters = agent._cluster_articles(articles, similarity)

    position = {a.url: i for i, a in enumerate(articles)}
    members = [[position[a.url] for a in cluster] for cluster in clusters]
    scores = clustering_scores(
        [row.story for row in dataset.articles],
        members,
    )
    return Result(
        name="clustering",
        metrics=scores.as_metrics(),
        detail={
            "config": agent.story_clustering_config.model_dump(
                include=set(OPTIONS),
            ),
            **subject.detail(),
            "clusters_found": len(clusters),
            "clusters_expected": len(dataset.stories()),
            "unclustered": len(articles) - sum(map(len, members)),
        },
    )
