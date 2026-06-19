# NewsBot

NewsBot monitors news sources, clusters related stories across outlets, and generates concise reports — delivered via email or browsable on the web.

**Live at [thenewsbot.net](https://thenewsbot.net)**

## What it does

- Fetches articles from RSS feeds on a configurable schedule
- Analyses and summarises articles using an LLM (Gemini or local via Ollama)
- Groups related stories across sources using text embeddings
- Produces HTML/Markdown reports and emails them to subscribers
- Exposes a web dashboard and REST API for managing configs and triggering runs

## Architecture

The system has three main parts:

| Component | Description |
|-----------|-------------|
| **NewsBot core** | Scraping, analysis, clustering, report generation |
| **Web interface** | Django app for browsing reports and managing configuration |
| **API** | FastAPI service for triggering runs programmatically |

Data is stored in PostgreSQL (or SQLite for local development).

## Docs

- [Development guide](docs/development.md) — local setup, environment variables, Docker, CLI reference
