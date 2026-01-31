"""Constants for the NewsBot project."""

from pytz import timezone

TIMEZONE_STR = "Europe/Vienna"
TZ = timezone(TIMEZONE_STR)

SENTIMENT_THRESHOLD = 0.2 # Threshold for sentiment classification
POLARITY_THRESHOLD = 0.1  # Threshold for polarity classification

# Minimum articles to form a story if fewer than config.min_sources
# sources
CLUSTER_SIZE_THRESHOLD = 3

# Fallback length for article summaries on error
MAX_ARTICLE_LENGTH_FALLBACK = 300
MAX_ARTICLE_CONTENT_LENGTH = 4000  # Max length of article content to process

# Max article content length for sentiment analysis (keeps under ~512
# tokens for pysentimiento)
SENTIMENT_MAX_CONTENT_LENGTH = 2048

# Batch size for embedding generation
# (optimized for Raspberry Pi 5 with 8GB RAM)
EMBEDDING_BATCH_SIZE = 16

# Batch size for geo location extraction using spaCy
# (optimized for Raspberry Pi 5 with 8GB RAM)
GEO_LOCATION_BATCH_SIZE = 10
