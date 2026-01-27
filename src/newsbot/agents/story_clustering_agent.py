"""
Story Clustering Agent.

Groups articles into stories and identifies top stories across sources
using semantic similarity, with optional geographical penalty to prevent
clustering articles about similar events in different locations.
"""

import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from newsbot.agents.judge_agent import JudgeAgent
from newsbot.agents.prompts import get_prompt
from newsbot.constants import (
    CLUSTER_SIZE_THRESHOLD,
    EMBEDDING_BATCH_SIZE,
    GEO_LOCATION_BATCH_SIZE,
)
from newsbot.llm_provider import get_llm_provider
from newsbot.model_cache import get_sentence_transformer, get_spacy_model
from newsbot.models import Article, Story
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

# Default geographical penalty when locations don't overlap
DEFAULT_GEO_PENALTY = 0.6


class StoryClusteringAgent:
    """
    Story Clustering Agent.

    Agent responsible for identifying and clustering stories across
    sources using semantic similarity.
    """

    def __init__(self, config: ConfigModel) -> None:
        """
        Initialize the Story Clustering Agent.

        Args:
            config: Configuration dictionary

        """
        self.story_clustering_config = config.story_clustering
        self.min_sources = self.story_clustering_config.min_sources
        self.similarity_threshold = (
            self.story_clustering_config.similarity_threshold
        )
        self.clustering_algorithm = self.story_clustering_config.algorithm
        self.dbscan_min_samples = (
            self.story_clustering_config.dbscan_min_samples
        )
        logger.info(
            f"Using clustering algorithm: {self.clustering_algorithm}",
        )
        if self.clustering_algorithm == "dbscan":
            logger.info(
                f"DBSCAN min_samples: {self.dbscan_min_samples}",
            )

        # Sampling configuration for limiting articles per cluster
        self.sampling_central_count = (
            self.story_clustering_config.sampling_central_count
        )
        self.sampling_recent_count = (
            self.story_clustering_config.sampling_recent_count
        )
        self.sampling_diverse_count = (
            self.story_clustering_config.sampling_diverse_count
        )
        self.sampling_similarity_floor = (
            self.story_clustering_config.sampling_similarity_floor
        )
        # Derive max_articles from the sum of sampling counts
        self.max_articles_per_story = (
            self.sampling_central_count
            + self.sampling_recent_count
            + self.sampling_diverse_count
        )
        logger.info(
            f"Sampling config: max_articles={self.max_articles_per_story} "
            f"(central={self.sampling_central_count}, "
            f"recent={self.sampling_recent_count}, "
            f"diverse={self.sampling_diverse_count}), "
            f"similarity_floor={self.sampling_similarity_floor}",
        )

        # Initialize sentence transformer model for semantic similarity
        # Uses global cache to reuse models across instances
        model_name = self.story_clustering_config.embedding_model
        try:
            self.embedding_model = get_sentence_transformer(model_name)
            logger.info("Sentence transformer model loaded successfully")
        except (OSError, RuntimeError):
            logger.exception("Error loading sentence transformer")
            logger.warning("Falling back to basic similarity")
            self.embedding_model = None

        # Initialize LLM provider for title generation
        self.provider = get_llm_provider(config)
        logger.info("Story clustering initialized with LLM provider")

        # Initialize judge agent for output validation
        self.judge_agent = JudgeAgent(config)
        self.llm_config = config.llm
        self.provider_name = self.llm_config.provider

        # Hybrid embeddings: use title + first 500 chars of content
        self.hybrid_content_chars = 500

        # Initialize spaCy multilingual NER for location extraction
        # Used for geographical penalty in clustering
        # Uses cached loader to reuse models across instances
        self.nlp = get_spacy_model("xx_ent_wiki_sm")
        logger.info("spaCy multilingual NER model loaded for geo penalty")

    def _get_embedding_texts(self, articles: list[Article]) -> list[str]:
        """
        Get texts for embedding using title + content.

        Combines article title with the first N characters of content
        for richer semantic signal, helping differentiate articles with
        similar headlines but different locations or contexts.

        Args:
            articles: List of articles to get embedding texts for.

        Returns:
            List of text strings to embed.

        """
        return [
            f"{article.title}. "
            f"{(article.content or '')[:self.hybrid_content_chars]}"
            for article in articles
        ]

    def _extract_locations(self, text: str) -> set[str]:
        """
        Extract location entities from text using spaCy NER.

        Extracts GPE (geo-political entities like countries, cities)
        and LOC (non-GPE locations like mountains, bodies of water).

        Args:
            text: Text to extract locations from.

        Returns:
            Set of location entity strings (lowercased for comparison).

        """
        if not text:
            return set()

        try:
            doc = self.nlp(text[:2000])  # Limit text length for performance
            locations = set()
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC"):
                    locations.add(ent.text.lower().strip())
        except (RuntimeError, ValueError) as e:
            logger.debug(f"Error extracting locations from text: {e}")
            return set()
        else:
            return locations

    def _get_article_locations(
        self,
        articles: list[Article],
    ) -> list[set[str]]:
        """
        Extract locations from all articles.

        Uses title and first portion of content for location extraction.
        Uses batch processing with spaCy's pipe() for better
        performance.

        Args:
            articles: List of articles to extract locations from.

        Returns:
            List of location sets, one per article.

        """
        if not articles:
            return []

        # Prepare texts for batch processing
        texts = [
            f"{article.title}. {(article.content or '')[:300]}"[:500]
            for article in articles
        ]
        try:
            docs = self.nlp.pipe(
                texts,
                batch_size=GEO_LOCATION_BATCH_SIZE,
                n_process=1,
            )

            # Stream docs to locations_list to reduce memory usage
            locations_list = []
            for doc in docs:
                locations_list.append({
                    ent.text.lower().strip()
                    for ent in doc.ents
                    if ent.label_ in ("GPE", "LOC")
                })
                del doc

        except (RuntimeError, ValueError, MemoryError):
            logger.exception(
                "Error in batch location extraction, "
                "falling back to sequential",
            )
            # Fallback to sequential processing if batch fails
            locations_list = []
            for text in texts:
                locations = self._extract_locations(text)
                locations_list.append(locations)

        # Log location extraction stats
        articles_with_locations = sum(1 for locs in locations_list if locs)
        logger.info(
            f"Extracted locations from {articles_with_locations}/"
            f"{len(articles)} articles",
        )

        return locations_list

    def _calculate_geo_penalty(
        self,
        locations_a: set[str],
        locations_b: set[str],
    ) -> float:
        """
        Calculate geographical penalty based on location overlap.

        Returns a factor between 0 and 1 to multiply with similarity.
        - 1.0: Locations overlap or one/both articles have no locations
        - DEFAULT_GEO_PENALTY: Locations are completely different

        Args:
            locations_a: Location entities from first article.
            locations_b: Location entities from second article.

        Returns:
            Penalty factor to apply to similarity score.

        """
        # No penalty if either article has no locations
        if not locations_a or not locations_b:
            return 1.0

        # Check for overlap
        overlap = locations_a & locations_b
        if overlap:
            return 1.0

        # Different locations - apply penalty
        return DEFAULT_GEO_PENALTY

    def _apply_geo_penalty_to_similarity(
        self,
        similarity_matrix: np.ndarray,
        article_locations: list[set[str]],
    ) -> np.ndarray:
        """
        Apply geographical penalty to similarity matrix.

        Multiplies each similarity score by the geo penalty factor
        based on location overlap between articles.

        Args:
            similarity_matrix: Original cosine similarity matrix.
            article_locations: List of location sets per article.

        Returns:
            Adjusted similarity matrix with geo penalties applied.

        """
        n = len(similarity_matrix)
        if n == 0:
            return similarity_matrix

        unique_locs = sorted(set().union(*article_locations))
        if not unique_locs:
            return similarity_matrix

        loc_to_idx = {loc: i for i, loc in enumerate(unique_locs)}

        # Build binary occurrence matrix L
        # L[i, j] = 1 if article i contains location j
        loc_matrix = np.zeros((n, len(unique_locs)), dtype=np.int8)
        rows: list[int] = []
        cols: list[int] = []
        for i, locs in enumerate(article_locations):
            for loc in locs:
                rows.append(i)
                cols.append(loc_to_idx[loc])
        if rows:
            loc_matrix[np.array(rows), np.array(cols)] = 1

        # Compute overlap matrix O = L * L.T
        # O[i, j] > 0 if articles i and j share at least one location
        overlap_matrix = loc_matrix @ loc_matrix.T

        # Identify articles that have NO locations (to exclude them from
        # penalty)
        has_locations = loc_matrix.sum(axis=1) > 0
        # Create a mask where penalty should be applied:
        # (Both have locations) AND (No overlap)
        # Use broadcasting to create an (n, n) mask of
        # "both have locations"
        both_have_locs = np.outer(has_locations, has_locations)
        penalty_mask = (overlap_matrix == 0) & both_have_locs

        # Apply penalty
        # We initialize a penalty matrix with 1.0 and fill masked areas
        # with DEFAULT_GEO_PENALTY
        final_penalties = np.ones((n, n))
        final_penalties[penalty_mask] = DEFAULT_GEO_PENALTY

        # Element-wise multiplication (Hadamard product)
        similarity_matrix *=  final_penalties

        logger.info(
            f"Applied vectorized geo penalty to {penalty_mask.sum() // 2} "
            "article pairs",
        )

        return similarity_matrix

    def _generate_story_title(self, cluster: list[Article]) -> str:
        """
        Generate a representative title for a story cluster using LLM.

        Args:
            cluster: list of articles in the story cluster

        Returns:
            Generated title string

        """
        if not cluster:
            logger.warning("Empty cluster provided for title generation")
            return "Untitled Story"

        try:
            # Collect all titles from the cluster
            titles = [article.title for article in cluster]
            sources = list({article.source for article in cluster})

            # Create prompt for LLM
            prompt_template = get_prompt(self.provider_name, "story_title.txt")
            prompt = prompt_template.format(
                num_titles=len(titles),
                num_sources=len(sources),
                titles_list=chr(10).join(f"- {title}" for title in titles),
            )

            # Generate title using provider
            generated_title = self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )

            # Clean up the title (remove quotes if LLM added them)
            generated_title = generated_title.strip('"').strip("'")

            # Validate and fix if needed
            prompt_context = (
                "A single, concise headline (10-12 words max) that captures "
                "the core story. ONLY the headline, no explanations or "
                "preambles."
            )
            generated_title = self.judge_agent.validate_and_fix(
                generated_title,
                prompt_context,
            )

            # Check if title is empty after validation (judge may return
            #  empty string after exhausting retries)
            if not generated_title or not generated_title.strip():
                fallback_title = max(
                    cluster,
                    key=lambda a: a.published_date,
                ).title
                logger.warning(
                    "LLM returned empty title after validation, "
                    f"using fallback: {fallback_title}",
                )
                return fallback_title

            logger.info(f"Generated story title: {generated_title}")

        except (RuntimeError, ValueError, ConnectionError):
            logger.exception("Error generating story title with LLM")
            # Fallback to most recent article's title
            fallback_title = max(cluster, key=lambda a: a.published_date).title
            logger.info(f"Using fallback title: {fallback_title}")
            return fallback_title
        else:
            return generated_title

    def _generate_titles_for_stories(self, stories: list[Story]) -> None:
        """
        Generate LLM titles for the given stories.

        Mutates stories in-place by replacing fallback titles with
        LLM-generated titles.

        Args:
            stories: list of Story objects to generate titles for

        """
        for story in stories:
            story.title = self._generate_story_title(story.articles)

    def identify_top_stories(
        self,
        articles: list[Article],
        top_n: int = 10,
    ) -> list[Story]:
        """
        Identify the top N stories from articles across sources.

        Uses semantic similarity.

        Args:
            articles: list of all scraped articles
            top_n: Number of top stories to return

        Returns:
            list of Story objects, sorted by number of sources covering
                them

        """
        logger.info(
            f"Identifying top {top_n} stories from {len(articles)} articles "
            f"using semantic similarity",
        )

        if self.embedding_model is None:
            logger.warning(
                "Embedding model not available, cannot perform semantic "
                "similarity clustering",
            )
            return []

        # Handle empty article list
        if not articles:
            logger.info("No articles to cluster")
            return []

        # Generate embeddings for articles (title + content)
        logger.info("Generating embeddings...")
        embedding_texts = self._get_embedding_texts(articles)
        embeddings = self.embedding_model.encode(
            embedding_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
        )

        # Calculate similarity matrix
        logger.info("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)

        # Apply geographical penalty to prevent clustering articles
        # about similar events in different locations
        logger.info("Extracting locations for geo penalty...")
        article_locations = self._get_article_locations(articles)
        similarity_matrix = self._apply_geo_penalty_to_similarity(
            similarity_matrix,
            article_locations,
        )

        # Group articles into clusters based on semantic similarity
        clusters = self._cluster_articles(articles, similarity_matrix)

        # Log clustering metrics
        self._log_clustering_metrics(clusters, articles)

        # Sample large clusters to limit articles per story
        article_to_idx = {a: i for i, a in enumerate(articles)}
        sampled_clusters = []
        for cluster in clusters:
            # Get embeddings for this cluster
            cluster_indices = [article_to_idx[a] for a in cluster]

            # Mean similarity of an article to all others in the cluster
            cluster_sim_slice = similarity_matrix[
                np.ix_(cluster_indices, cluster_indices)
            ]
            centrality_scores = cluster_sim_slice.mean(axis=1)


            # Map centrality back to global article mapping for the
            # sampler
            centrality_map = {
                articles[idx]: score
                for idx, score in zip(
                    cluster_indices, centrality_scores, strict=True,
                )
            }

            # Sample articles using the global centrality map
            sampled_cluster = self._sample_cluster_articles_stratified(
                cluster,
                similarity_matrix,
                centrality_map,
                article_to_idx,
            )
            sampled_clusters.append(sampled_cluster)

        clusters = sampled_clusters

        # Convert clusters to Story objects
        stories = self._clusters_to_stories(clusters)

        # Sort by number of sources (descending), then by article count
        stories.sort(
            key=lambda s: (len(s.sources), s.article_count),
            reverse=True,
        )

        # Return top N stories
        top_stories = stories[:top_n]

        # Generate LLM titles only for the selected top stories
        self._generate_titles_for_stories(top_stories)

        logger.info(f"Top {len(top_stories)} stories identified")
        for i, story in enumerate(top_stories, 1):
            logger.info(
                f"{i}. '{story.title[:60]}...' - "
                f"{len(story.sources)} sources, "
                f"{story.article_count} articles",
            )

        return top_stories

    def _cluster_articles(
        self,
        articles: list[Article],
        similarity_matrix: np.ndarray,
    ) -> list[list[Article]]:
        """
        Cluster articles based on similarity matrix.

        Dispatches to the appropriate clustering algorithm.

        Args:
            articles: list of articles to cluster
            similarity_matrix: Pre-computed similarity matrix

        Returns:
            list of article clusters

        """
        if self.clustering_algorithm == "dbscan":
            return self._cluster_articles_dbscan(articles, similarity_matrix)

        return self._cluster_articles_greedy(articles, similarity_matrix)

    def _cluster_articles_greedy(
        self,
        articles: list[Article],
        similarity_matrix: np.ndarray,
    ) -> list[list[Article]]:
        """
        Cluster articles using greedy single-linkage algorithm.

        Args:
            articles: list of articles to cluster
            similarity_matrix: Pre-computed similarity matrix

        Returns:
            list of article clusters

        """
        clusters = []
        processed = set()

        for i, article1 in enumerate(articles):
            if i in processed:
                continue

            # Start a new cluster
            cluster = [article1]
            cluster_indices = [i]
            processed.add(i)

            # Find similar articles
            for j in range(i + 1, len(articles)):
                if j in processed:
                    continue

                # Check similarity with any article in the current
                # cluster
                max_similarity = float(
                    np.max(
                        similarity_matrix[np.asarray(cluster_indices), j],
                    ),
                )

                if max_similarity >= self.similarity_threshold:
                    cluster.append(articles[j])
                    cluster_indices.append(j)
                    processed.add(j)

            if len(cluster) > 0:
                clusters.append(cluster)

        logger.info(
            f"Identified {len(clusters)} potential story clusters using "
            "greedy single-linkage clustering",
        )

        return clusters

    def _cluster_articles_dbscan(
        self,
        articles: list[Article],
        similarity_matrix: np.ndarray,
    ) -> list[list[Article]]:
        """
        Cluster articles using DBSCAN algorithm.

        Args:
            articles: list of articles to cluster
            similarity_matrix: Pre-computed similarity matrix

        Returns:
            list of article clusters

        """
        # Convert similarity matrix to distance matrix for DBSCAN
        similarity_array = np.array(similarity_matrix)

        # Clamp similarity values to [0, 1] to avoid negative distances
        # due to floating point issues
        similarity_array = np.clip(similarity_array, 0.0, 1.0)

        # Get distance matrix
        distance_matrix = 1 - similarity_array

        # Ensure non-negative distances (should already be, but
        # double-check)
        distance_matrix = np.clip(distance_matrix, 0.0, None)

        # Convert similarity threshold to distance epsilon
        eps = 1 - self.similarity_threshold

        # Run DBSCAN with precomputed distance matrix
        dbscan = DBSCAN(
            eps=eps,
            min_samples=self.dbscan_min_samples,
            metric="precomputed",
        )
        labels = dbscan.fit_predict(distance_matrix)

        # Group articles by cluster label
        noise_count = int((labels == -1).sum())
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        cluster_list: list[list[Article]] = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_list.append([articles[i] for i in indices])

        logger.info(
            f"Identified {len(cluster_list)} potential story clusters using "
            f"DBSCAN clustering ({noise_count} noise points excluded)",
        )

        return cluster_list

    def _log_clustering_metrics(
        self,
        clusters: list[list[Article]],
        all_articles: list[Article],
    ) -> None:
        """
        Log clustering metrics for observability.

        Args:
            clusters: list of article clusters
            all_articles: all articles that were clustered

        """
        if not clusters:
            logger.info("No clusters found")
            return

        cluster_sizes = np.array([len(cluster) for cluster in clusters])
        total_clustered = int(cluster_sizes.sum())
        noise_count = len(all_articles) - total_clustered

        avg_cluster_size = float(cluster_sizes.mean())
        max_cluster_size = int(cluster_sizes.max())
        min_cluster_size = int(cluster_sizes.min())

        # Count sources per cluster
        sources_per_cluster = np.array(
            [
                len({article.source for article in cluster})
                for cluster in clusters
            ],
        )
        avg_sources = float(sources_per_cluster.mean())

        logger.info(
            f"Clustering metrics: {len(clusters)} clusters, "
            f"{total_clustered} articles clustered, "
            f"{noise_count} noise/outlier articles",
        )
        logger.info(
            f"Cluster size: avg={avg_cluster_size:.1f}, "
            f"min={min_cluster_size}, max={max_cluster_size}",
        )
        logger.info(
            f"Sources per cluster: avg={avg_sources:.1f}",
        )

    def _clusters_to_stories(
        self,
        clusters: list[list[Article]],
    ) -> list[Story]:
        """
        Convert article clusters to Story objects.

        Uses fallback titles (most recent article's title) for initial
        creation. LLM titles are generated later for selected stories.

        Args:
            clusters: list of article clusters

        Returns:
            list of Story objects

        """
        stories = []
        for i, cluster in enumerate(clusters):
            # Get unique sources in this cluster
            cluster_sources = list({article.source for article in cluster})

            # Only include stories covered by multiple sources (or
            # single source with multiple articles)
            if (
                len(cluster_sources) >= self.min_sources
                or len(cluster) >= CLUSTER_SIZE_THRESHOLD
            ):
                # Use fallback title (most recent article's title)
                # LLM titles are generated later for top N stories only
                fallback_title = max(
                    cluster,
                    key=lambda a: a.published_date,
                ).title

                story = Story(
                    story_id=f"story_{i + 1}",
                    title=fallback_title,
                    articles=cluster,
                    sources=cluster_sources,
                    article_count=len(cluster),
                    earliest_date=min(
                        article.published_date for article in cluster
                    ),
                    latest_date=max(
                        article.published_date for article in cluster
                    ),
                )
                stories.append(story)

        return stories

    def _sample_cluster_articles_stratified(
        self,
        cluster: list[Article],
        similarity_matrix: np.ndarray,
        centrality_map: dict[Article, float],
        article_to_idx: dict[Article, int],
    ) -> list[Article]:
        """
        Sample articles from a cluster using stratified selection.

        Selects articles in three categories:
        1. Central: highest similarity to cluster centroid
        2. Recent: most recently published
        3. Diverse: MMR selection (most dissimilar to already-selected)

        Source diversity is enforced within each category.

        Args:
            cluster: Articles in the cluster.
            similarity_matrix: Pairwise similarity matrix for cluster
                articles (row/col indices match article_to_idx).
            centrality_map: Pre-computed similarity to centroid for
                each article.
            article_to_idx: Mapping from article to matrix index.

        Returns:
            Sampled list of articles, or original cluster if below
            threshold.

        """
        if len(cluster) <= self.max_articles_per_story:
            return cluster

        logger.info(
            f"Sampling cluster of {len(cluster)} articles down to "
            f"{self.max_articles_per_story}",
        )

        selected: list[Article] = []
        remaining = list(cluster)

        # 1. CENTRAL
        central_ranked = sorted(
            remaining,
            key=lambda a: centrality_map[a],
            reverse=True,
        )
        central_selected = self._select_with_source_diversity(
            central_ranked, self.sampling_central_count,
        )
        selected.extend(central_selected)
        remaining = [a for a in remaining if a not in selected]

        # 2. RECENT
        recent_ranked = sorted(
            remaining,
            key=lambda a: a.published_date,
            reverse=True,
        )
        recent_selected = self._select_with_source_diversity(
            recent_ranked, self.sampling_recent_count,
        )
        selected.extend(recent_selected)
        remaining = [a for a in remaining if a not in selected]

        # 3. DIVERSE (MMR)
        diverse_selected = self._select_diverse_mmr(
            remaining,
            selected,
            similarity_matrix,
            centrality_map,
            article_to_idx,
            self.sampling_diverse_count,
        )
        selected.extend(diverse_selected)

        return selected

    def _select_with_source_diversity(
        self,
        ranked_articles: list[Article],
        n: int,
    ) -> list[Article]:
        """
        Select top N articles while preferring source diversity.

        First pass selects one article per source (in rank order).
        Second pass fills remaining slots with highest-ranked articles.

        Args:
            ranked_articles: Articles pre-sorted by desired ranking.
            n: Number of articles to select.

        Returns:
            Selected articles with source diversity.

        """
        selected: list[Article] = []
        sources_seen: set[str] = set()

        # First pass: one per source
        for article in ranked_articles:
            if article.source not in sources_seen:
                selected.append(article)
                sources_seen.add(article.source)
                if len(selected) >= n:
                    return selected

        # Second pass: fill remaining
        for article in ranked_articles:
            if article not in selected:
                selected.append(article)
                if len(selected) >= n:
                    return selected

        return selected

    def _select_diverse_mmr(
        self,
        remaining: list[Article],
        selected: list[Article],
        similarity_matrix: np.ndarray,
        centrality_map: dict[Article, float],
        article_to_idx: dict[Article, int],
        n: int,
    ) -> list[Article]:
        """
        Select articles maximizing diversity via MMR.

        Iteratively selects articles that are most dissimilar to the
        already-selected set, while respecting a minimum similarity
        floor to the cluster centroid.

        Args:
            remaining: Articles not yet selected.
            selected: Articles already selected.
            similarity_matrix: Pairwise similarity matrix (indices
                match article_to_idx).
            centrality_map: Similarity to centroid for each article.
            article_to_idx: Mapping from article to matrix index.
            n: Number of articles to select.

        Returns:
            Selected diverse articles.

        """
        diverse: list[Article] = []
        selected_indices: list[int] = [article_to_idx[a] for a in selected]
        remaining_list = list(remaining)
        floor = self.sampling_similarity_floor

        for _ in range(n):
            if not remaining_list:
                break

            remaining_indices = np.array(
                [article_to_idx[a] for a in remaining_list],
                dtype=np.intp,
            )
            centrality = np.array(
                [centrality_map[a] for a in remaining_list],
                dtype=np.float64,
            )
            floor_ok = centrality >= floor

            if selected_indices:
                sel = np.asarray(selected_indices, dtype=np.intp)
                sim_slice = similarity_matrix[
                    np.ix_(remaining_indices, sel)
                ]
                max_sim = np.max(sim_slice, axis=1)
            else:
                max_sim = np.zeros(len(remaining_list), dtype=np.float64)

            max_sim = np.where(floor_ok, max_sim, np.inf)
            if np.all(~np.isfinite(max_sim)):
                break

            best_local = int(np.argmin(max_sim))
            best_article = remaining_list[best_local]
            diverse.append(best_article)
            selected_indices.append(article_to_idx[best_article])
            remaining_list.pop(best_local)

        return diverse
