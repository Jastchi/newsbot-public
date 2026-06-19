# Development Guide

## Prerequisites

- **[uv](https://github.com/astral-sh/uv)** — Fast Python package installer and runner
- **[ollama](https://ollama.ai)** — For local LLM inference (optional)
- **[Docker](https://www.docker.com/)** — For containerized deployment (optional)

## Local Setup

```bash
git clone https://github.com/JastchiLabs/newsbot.git
cd newsbot
uv sync
uv run src/web/manage.py migrate
```

**Pre-commit hooks (recommended):**
```bash
uv run pre-commit install
uv run playwright install --with-deps chromium  # for screenshot tests
```

Hooks run `ty check`, `ruff check`, and enforce no inline ignore comments before each commit. Rule ignores are managed in `pyproject.toml`.

## Environment Variables

Create a `.env` file in the project root:

```env
# Database (PostgreSQL/Supabase — omit for SQLite)
DATABASE_URL=postgresql://postgres.user:password@host:5432/postgres
SUPABASE_SERVICE_KEY=your-service-role-key-here

# Email
EMAIL_ENABLED=true
EMAIL_PROVIDER=smtp
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USE_SSL=false
EMAIL_SENDER=yourbot@yourdomain.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENT=admin@yourdomain.com
EMAIL_FOR_CANCELLATION=cancellation@yourdomain.com
NEWSSERVER_BASE_URL=https://yourdomain.com

# LLM (only when using Gemini)
GEMINI_API_KEY=your-gemini-api-key

# Django (production)
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=false
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com
DJANGO_CSRF_TRUSTED_ORIGINS=localhost:8000,yourdomain.com
```

## Database Setup

**Option A — Supabase:**
1. Create a project at https://supabase.com
2. Get the PostgreSQL connection string from Project Settings → Database
3. Get the Service Role key from Project Settings → API
4. Set `DATABASE_URL` and `SUPABASE_SERVICE_KEY` in `.env`

**Option B — SQLite:**  
Leave `DATABASE_URL` unset — SQLite databases are created automatically.

## Running Locally

**Web interface:**
```bash
uv run ./src/web/manage.py runserver 8001
# open http://localhost:8001
```

**API server:**
```bash
uv run python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/docs
```

**NewsBot CLI:**
```bash
uv run newsbot run --config config_key        # scrape + analyse
uv run newsbot analyze --config config_key    # analyse existing data
uv run newsbot schedule --config config_key   # run on schedule
```

Options: `--force`, `--email-receivers [addr ...]`, `--test`

## Dependency Groups

| Group | Description |
|-------|-------------|
| `newsbot` | Core scraping, LLMs, sentiment, clustering |
| `serve` | Django production server (gunicorn, whitenoise) |
| `api` | FastAPI-based REST API (includes `newsbot`) |
| `raspberrypi` | Raspberry Pi deployment (includes `newsbot`) |
| `test` | pytest, playwright, httpx |
| `dev` | linting, type checking, pre-commit |

`uv sync` installs all groups by default. To install selectively:

| Scenario | Command |
|----------|---------|
| Web server (production) | `uv sync --no-default-groups --group serve` |
| API server (production) | `uv sync --no-default-groups --group api` |
| Raspberry Pi | `uv sync --no-default-groups --group raspberrypi` |
| Tests | `uv sync --no-default-groups --group test --group <target>` |

## Docker Deployment

```bash
# Build
docker build -f Dockerfile.api -t newsbot-api .
docker build -f Dockerfile.web -t newsbot-web .

# Run
docker run -d --name newsbot-api -p 8000:8000 --env-file .env newsbot-api
docker run -d --name newsbot-web -p 8001:8000 --env-file .env newsbot-web

# Stop
docker stop newsbot-api newsbot-web && docker rm newsbot-api newsbot-web
```

Enable the built-in scheduler in the API container:
```bash
docker run -d --name newsbot-api -p 8000:8000 -e ENABLE_SCHEDULER=true --env-file .env newsbot-api
```

When `ENABLE_SCHEDULER=true`, schedules refresh daily at 00:05 Europe/Vienna. Not suitable for serverless platforms — use an external scheduler to call the API endpoints directly instead.

## API Reference

**Health check:** `GET /health`

**Trigger scrape:** `POST /run/{config_key}` — body: `{"force": false}`

**Trigger analysis:** `POST /analyze/{config_key}` — body: `{"days": 7, "test": false, "email_receivers": []}`

**Get schedules:** `GET /schedules`

Full interactive docs at `/docs` when the API server is running.

## Email Configuration

Supports **SMTP** (default) and **EmailJS**.

**SMTP:** Set `EMAIL_PROVIDER=smtp` and SMTP variables in `.env`. Manage subscribers via Django admin (`/admin/`).

**EmailJS:** Set `EMAIL_PROVIDER=emailjs` and create a template with variables `{{subject}}`, `{{{content}}}`, `{{to_email}}`, `{{bcc}}`, `{{from_name}}`, `{{from_email}}`, `{{sender_name}}`, `{{topic}}`. Set `EMAILJS_SERVICE_ID`, `EMAILJS_TEMPLATE_ID`, `EMAILJS_USER_ID`, `EMAIL_SENDER` in `.env`.

Note: For a custom sender display name, SMTP is more reliable than EmailJS with personal email providers.
