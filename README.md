# NewsBot

An automated news monitoring and analysis system that scrapes news sources, performs sentiment analysis, clusters related stories, and generates comprehensive reports.

## Overview

NewsBot is a modular news analysis pipeline with three main components:

1. **NewsBot Core** - The main news scraping and analysis engine
2. **Web Interface** - A Django web application for viewing and managing reports
3. **API** - RESTful API for triggering scraping and analysis operations
4. **After-Analysis Hooks** - Post-processing scripts (e.g., email notifications)

The system now supports:
- **External PostgreSQL Database** - Centralized configuration and data storage
- **Docker Containers** - Easy deployment with isolated API and Web services
- **Cloud-Ready Architecture** - Designed for serverless production deployment

## Features

- 📰 **RSS Feed Scraping** - Fetches articles from configured news sources
- 🤖 **Analysis** - Uses local LLM (via Ollama) or cloud LLM (Gemini) for summarisation and analysis
- ✅ **Name Validation** - Validates named entities in LLM output against source articles to prevent hallucinated names
- 😊 **Sentiment Analysis** - Evaluates article sentiment using PySentimiento/TextBlob/VADER
- 🔗 **Story Clustering** - Groups related articles across different sources using DBSCAN or greedy algorithm on text embeddings
- 📊 **Report Generation** - Creates HTML/Markdown reports of top stories
- 📧 **Email Notifications** - Sends reports via email with subscriber management
- 🗄️ **PostgreSQL Database** - Centralized data storage
- ⏰ **Scheduled Execution** - Automatic daily runs with APScheduler
- 🌐 **Web Dashboard** - View, manage, and browse generated reports and report configurations
- 🐳 **Docker Support** - Production-ready containerization for API and Web services
- 📡 **RESTful API** - Trigger operations programmatically and integrate with cloud schedulers

## Prerequisites

- **[uv](https://github.com/astral-sh/uv)** - Fast Python package installer and runner (if not using Docker)
- **[ollama](https://ollama.ai)** - For local LLM inference (if needed)
- **[Docker](https://www.docker.com/)** - For containerized deployment (if needed)

## Installation

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jastchi/newsbot.git
   cd newsbot
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Initialize database migrations:**
   ```bash
   uv run src/web/manage.py migrate
   ```

4. **Install pre-commit hooks and playwright (optional but recommended):**
   ```bash
   # Check code quality before every commit
   uv run pre-commit install

   # For screenshot tests - will fail if not installed
   uv run playwright install --with-deps chromium
   ```
   
   This installs git hooks that automatically run type checking (`ty check`), linting (`ruff check`), and checks for ignore comments (which are not allowed in this repository - rule ignores should be managed in `pyproject.toml`) before each commit.

### Docker Setup (Production)

Build and run the API and Web services as individual containers.

**1. Build Docker Images:**

```bash
# Build API image
docker build -f Dockerfile.api -t newsbot-api .

# Build Web image
docker build -f Dockerfile.web -t newsbot-web .
```

**2. Run Containers:**

```bash
# Run API container
docker run -d \
  --name newsbot-api \
  -p 8000:8000 \
  --env-file .env 
  newsbot-api

# Run Web container
docker run -d \
  --name newsbot-web \
  -p 8001:8000 \
  --env-file .env \
  newsbot-web
```

**3. Access Services:**

- **API Docs:** http://localhost:8000/docs
- **Web Dashboard:** http://localhost:8001
- **Admin Interface:** http://localhost:8001/admin/

Both containers connect to the external PostgreSQL database specified in your `.env` file.

**Stop Containers:**
```bash
docker stop newsbot-api newsbot-web
docker rm newsbot-api newsbot-web
```

## Dependency Groups

The project uses uv dependency groups to manage dependencies for different deployment scenarios.

By default, `uv sync` installs **all** groups (configured via `default-groups = "all"` in pyproject.toml). To install only specific groups, use `--no-default-groups --group <group>`.

| Group | Description | Includes |
|-------|-------------|----------|
| `newsbot` | Core newsbot functionality (scraping, LLMs, sentiment analysis, clustering) | - |
| `serve` | Django production server (gunicorn, whitenoise) | - |
| `api` | FastAPI-based REST API | `newsbot` |
| `raspberrypi` | Raspberry Pi deployment with file watcher | `newsbot` |
| `test` | Testing dependencies (pytest, playwright, httpx) | - |
| `dev` | Development tools (linting, type checking, pre-commit) | - |

### Deployment Scenarios

| Scenario | Command |
|----------|---------|
| **Local development** | `uv sync` (installs all groups) |
| **NewsBot CLI only** | `uv sync --no-default-groups --group newsbot` |
| **Web server (production)** | `uv sync --no-default-groups --group serve` |
| **API server (production)** | `uv sync --no-default-groups --group api` |
| **Raspberry Pi** | `uv sync --no-default-groups --group raspberrypi` |
| **Running tests** | `uv sync --no-default-groups --group test --group <target>` |

### Docker Images

- **Dockerfile.api** (test stage): `--group test --group api`
- **Dockerfile.api** (production stage): `--group serve`
- **Dockerfile.web** (test stage): `--group test --group serve`
- **Dockerfile.web** (production stage): `--group serve`

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Database Configuration (PostgreSQL/Supabase)
DATABASE_URL=postgresql://postgres.user:password@host:5432/postgres
SUPABASE_SERVICE_KEY=your-service-role-key-here # For report storage - only if using Supabase DB

# Email Configuration (for error handling and report sending)
EMAIL_ENABLED=true
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USE_SSL=false
EMAIL_SENDER=yourbot@yourdomain.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENT=admin@yourdomain.com
EMAIL_FOR_CANCELLATION=cancellation@yourdomain.com

# Django Web Interface (only needed in Production)
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=false  # Set to true only for development
FORCE_SCRIPT_NAME=app-root  # If running under localhost/<app-root>
STATIC_ROOT=staticfiles-folder-path

# CORS/Security Settings
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com
DJANGO_CSRF_TRUSTED_ORIGINS=localhost:8000,yourdomain.com


or add them to the environment of the docker containers.

### 2. Database Setup

NewsBot requires SQLite or a cloud PostgreSQL database for storing configurations and data. 

**Option A: Using Supabase**

1. Create a Supabase project at https://supabase.com
2. Get your PostgreSQL connection string from Project Settings → Database
3. Get your Service Role key from Project Settings → API
4. Set these in your `.env`:
   ```env
   DATABASE_URL=postgresql://postgres.xxxxx:password@aws-0-region.pooler.supabase.com:5432/postgres
   SUPABASE_SERVICE_KEY=your-service-role-key
   ```

**Option B: Self-Hosted SQLite**

Do not set a DATABASE_URL - SQLite databases will automatically be created

### 3. News Configuration

News sources, LLM settings, and cluster parameters are stored in the database. Configure them through the web interface, as admin.

- **Django Management Command**:
  ```bash
  uv run src/web/manage.py createsuperuser
  ```

Report subscribers are also managed here.

## Usage

### NewsBot Core CLI

Run newsbot operations from the command line:

#### Run Once (Immediate Execution)
```bash
uv run newsbot run --config config_key
```

#### Analyze Existing Data
```bash
uv run newsbot analyze --config config_key
```

#### Scheduled Mode (Daily Runs)
```bash
uv run newsbot schedule --config config_key
```

**Options:**
- `--config` - Config key from database (must exist in NewsConfig)
- `--force` - Force re-scraping even if already run today
- `--email-receivers` - Override email recipients
  - No arguments: Send only to sender
  - Email addresses: Send to specified addresses instead of database recipients
- `--test` - Replace articles with test data before analysis

**Examples:**
```bash
# Scrape sources (default)
uv run newsbot run --config config_key

# Force re-scrape even if done today
uv run newsbot run --config config_key --force

# Use database email recipients
uv run newsbot analyze --config config_key

# Send email only to sender
uv run newsbot analyze --config config_key --email-receivers

# Send to specific emails
uv run newsbot analyze --config technology --email-receivers user1@example.com user2@example.com

# Use test dataset
uv run newsbot analyze --config technology --test
```

Generated reports are saved to `reports/<config_key>/` and to supabase storage, if set up.

### 2. Web Interface

The Django web application lets you manage configs, view reports, and monitor runs.

**Start Development Server:**
```bash
cd src/web
uv run ./src/web/manage.py runserver 8001
```

Open http://localhost:8001 in your browser.

**Admin Interface:**
- URL: http://localhost:8000/admin/
- Create a superuser: `uv run ./src/web/manage.py createsuperuser`
- Manage news configs, sources, and email subscribers

**Main Views:**
- **Configs** - Overview of all news configurations with latest reports
- **Runs** - View scraping and analysis history
- **Logs** - Real-time log viewer with streaming support
- **Admin** - Manage all configurations and subscribers

**Production Deployment:**
```bash
uv run gunicorn --bind 0.0.0.0:9000 web.web.wsgi
```

### 3. RESTful API

The FastAPI-based API allows programmatic control and integration with cloud schedulers (e.g. via [GitHub](https://github.com/) workflow).

**Start API Server:**
```bash
uv run python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs in your browser for interactive API documentation.

#### API Endpoints

**Health Check**
```bash
GET /health
```

**Start Daily Scrape** (runs in background)
```bash
POST /run/{config_key}
Content-Type: application/json

{
  "force": false
}
```

Response:
```json
{
  "status": "started",
  "message": "Daily scrape started for config 'config_key'",
  "config_key": "config_key",
  "config_name": "News"
}
```

**Start Weekly Analysis** (runs in background)
```bash
POST /analyze/{config_key}
Content-Type: application/json

{
  "days": 7,
  "test": false,
  "email_receivers": ["user@example.com"]
}
```

Response:
```json
{
  "status": "started",
  "message": "Weekly analysis started for config 'config_key'",
  "config_key": "config_key",
  "config_name": "News"
}
```

**Get All Schedules** (for cloud scheduler integration)
```bash
GET /schedules
```

Response:
```json
{
  "configs": [
    {
      "key": "config_key1",
      "name": "News",
      "daily_scrape": {
        "enabled": true,
        "hour": 2,
        "minute": 0,
        "cron": "0 2 * * *"
      },
      "weekly_analysis": {
        "enabled": true,
        "day_of_week": "MON",
        "hour": 9,
        "minute": 0,
        "cron": "0 9 ? * MON"
      }
    }
  ]
}
```

#### API Error Responses

**Config Not Found (404)**
```json
{
  "success": false,
  "error": "Config 'invalid_key' not found",
  "errors": []
}
```

**Server Error (500)**
```json
{
  "success": false,
  "error": "Internal server error",
  "errors": ["Error details here"]
}
```

#### Using the Scheduler Script

You may use the scheduler script to automatically run scheduled tasks via the API. The scheduler runs as a service, fetches schedules from the API at 00:05 daily, and triggers API endpoints when tasks are due.

**Start the Scheduler:**
```bash
# Run with default settings (connects to localhost:8000)
uv run scripts/scheduler.py

# Run with custom port
uv run python scripts/scheduler.py --port 8080

# Run with custom host and port
uv run python scripts/scheduler.py --host api.example.com --port 443
```

**Requirements:**
- The API server must be running and accessible

**Options:**
- `--host` - API server host (default: localhost)
- `--port` - API server port (default: 8000)
- `--refresh-hour` - Hour to refresh schedules (default: 0)
- `--refresh-minute` - Minute to refresh schedules (default: 5)

### 4. Docker Deployment

The project includes two Dockerfiles for containerized deployment, see [Docker Setup](#docker-setup-production).

#### Enabling the Scheduler in Docker

The API container supports an optional built-in scheduler that can be enabled via the `ENABLE_SCHEDULER` environment variable:

```bash
# Run API container with scheduler enabled
docker run -d \
  --name newsbot-api \
  -p 8000:8000 \
  -e ENABLE_SCHEDULER=true \
  --env-file .env \
  newsbot-api
```

When `ENABLE_SCHEDULER=true`:
- The scheduler service starts in the background
- Schedules are refreshed daily at 00:05 Europe/Vienna
- Tasks are triggered at their configured times

**Alternatively**, you can run the scheduler as a separate service or use a cloud scheduler to call the API endpoints directly.

**Note:** The scheduler runs as a long-lived blocking process and is not suitable for serverless platforms that expect short-lived containers. For serverless deployments, use an external scheduler to call the API endpoints directly.

### 5. After-Analysis Hooks

#### Email Sender

Sends generated reports via email. Recipients are managed through the Django web interface.

**Setup:**

1. Configure SMTP settings in environment variables (see [Configuration](#configuration))
2. Create superuser: `uv run src/web/manage.py createsuperuser`
3. Log in to Django admin: http://localhost:8000/admin/
4. Add email subscribers in "Subscribers" section
5. Assign configs to each subscriber

The email hook automatically triggers after report generation when enabled.
