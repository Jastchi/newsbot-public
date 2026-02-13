"""Constants for the NewsBot project."""

from pytz import timezone

TIMEZONE_STR = "Europe/Vienna"
TZ = timezone(TIMEZONE_STR)

# Daily scrape / config refresh time (hour 0-23, minute 0-59)
DAILY_SCRAPE_HOUR = 0
DAILY_SCRAPE_MINUTE = 5

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

# --- Day-of-week mappings (single source of truth for API and web) ---
# Python weekday: Monday=0, Sunday=6 (same as datetime.date.weekday())
DAY_NAME_TO_PYTHON_WEEKDAY: dict[str, int] = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}
PYTHON_WEEKDAY_TO_DAY_NAME: dict[int, str] = {
    v: k for k, v in DAY_NAME_TO_PYTHON_WEEKDAY.items()
}
# Cron convention: Sunday=0, Monday=1, ..., Saturday=6
DAY_NAME_TO_CRON_WEEKDAY: dict[str, int] = {
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6,
}
