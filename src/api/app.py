"""
FastAPI application for newsbot API.

Provides POST endpoints for triggering newsbot operations.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.handlers import get_all_schedules, handle_analyze, handle_run
from api.job_manager import Job, JobStatus, JobType, job_manager
from newsbot.constants import TZ
from utilities import ConfigNotFoundError, load_config, setup_django

logger = logging.getLogger(__name__)


# Request/Response models
class RunRequest(BaseModel):
    """Request body for /run/{config_key} endpoint."""

    force: bool = Field(
        default=False,
        description="Force scrape even if already done today",
    )


class AnalyzeRequest(BaseModel):
    """Request body for /analyze/{config_key} endpoint."""

    days: int | None = Field(
        default=None,
        description="Number of days to analyze (defaults to config value)",
    )
    test: bool = Field(
        default=False,
        description="Use test articles (only for test configs)",
    )
    email_receivers: list[str] | None = Field(
        default=None,
        description="Optional list of email receivers to override",
    )


class JobStartResponse(BaseModel):
    """Response model for starting background jobs."""

    job_id: str
    status: str
    message: str
    config_key: str
    config_name: str


class ScheduleConfig(BaseModel):
    """Schedule configuration for a single config."""

    key: str
    name: str
    is_active: bool
    daily_scrape: dict[str, Any]
    weekly_analysis: dict[str, Any]


class SchedulesResponse(BaseModel):
    """Response model for /schedules endpoint."""

    configs: list[ScheduleConfig]


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    message: str
    scheduler_enabled: bool
    scheduler_running: bool
    scheduler_pid: int | None = None


class ErrorResponse(BaseModel):
    """Response model for error responses."""

    success: bool = False
    error: str
    errors: list[str] = Field(default_factory=list)


class ScheduledJob(BaseModel):
    """Model representing a job that is scheduled to run today."""

    type: JobType
    config_key: str
    config_name: str
    scheduled_at: str  # "HH:MM"


class TodayJobsResponse(BaseModel):
    """Response model for /jobs/today endpoint."""

    realized_jobs: list[Job]
    scheduled_jobs: list[ScheduledJob]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Initializes Django at startup.
    """
    # Startup: Initialize Django
    logger.info("Initializing Django...")
    setup_django()
    logger.info("Django initialized successfully")

    yield

    # Shutdown: cleanup if needed
    logger.info("API shutting down")


# Create FastAPI application
app = FastAPI(
    title="NewsBot API",
    description="API for triggering newsbot operations",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_scheduler_status() -> tuple[bool, int | None]:
    """Check if the scheduler process is running."""
    scheduler_pid = os.environ.get("SCHEDULER_PID")
    if scheduler_pid is None:
        return False, None

    pid = int(scheduler_pid)
    try:
        # Check if process is still running (signal 0)
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False, None
    else:
        return True, pid


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    scheduler_enabled = os.getenv("ENABLE_SCHEDULER", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    scheduler_running, scheduler_pid = (
        _get_scheduler_status() if scheduler_enabled else (False, None)
    )

    return HealthResponse(
        status="ok",
        message="NewsBot API is running",
        scheduler_enabled=scheduler_enabled,
        scheduler_running=scheduler_running,
        scheduler_pid=scheduler_pid,
    )


@app.post(
    "/run/{config_key}",
    response_model=JobStartResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Config not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def run_scrape(
    config_key: str,
    background_tasks: BackgroundTasks,
    request: RunRequest | None = None,
) -> JobStartResponse:
    """
    Start daily scrape for a configuration in the background.

    Args:
        config_key: Configuration key from database (e.g., "technology")
        background_tasks: FastAPI background tasks manager
        request: Optional request body with force flag

    Returns:
        RunResponse with status indicating the task has started

    """
    # Validate config exists before starting background task
    try:
        config, _ = load_config(config_key)
    except ConfigNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Error loading config '{config_key}'")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Parse request body (use defaults if not provided)
    force = request.force if request else False

    # Create job tracker
    job_id = job_manager.create_job(JobType.SCRAPE, config_key)

    # Add handler to background tasks
    background_tasks.add_task(
        handle_run,
        config_key=config_key,
        force=force,
        job_id=job_id,
    )

    # Return immediately
    return JobStartResponse(
        job_id=job_id,
        status="pending",
        message=f"Daily scrape started for config '{config_key}'",
        config_key=config_key,
        config_name=config.name,
    )


@app.post(
    "/analyze/{config_key}",
    response_model=JobStartResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Config not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def run_analysis(
    config_key: str,
    background_tasks: BackgroundTasks,
    request: AnalyzeRequest | None = None,
) -> JobStartResponse:
    """
    Start weekly analysis for a configuration in the background.

    Args:
        config_key: Configuration key from database (e.g., "technology")
        background_tasks: FastAPI background tasks manager
        request: Optional request body with days, test, and
            email_receivers

    Returns:
        AnalyzeResponse with status indicating the task has started

    """
    # Validate config exists before starting background task
    try:
        config, _ = load_config(config_key)
    except ConfigNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Error loading config '{config_key}'")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Parse request body (use defaults if not provided)
    days = request.days if request else None
    test = request.test if request else False
    email_receivers = request.email_receivers if request else None

    # Create job tracker
    job_id = job_manager.create_job(JobType.ANALYSIS, config_key)

    # Add handler to background tasks
    background_tasks.add_task(
        handle_analyze,
        config_key=config_key,
        days=days,
        test=test,
        email_receivers=email_receivers,
        job_id=job_id,
    )

    # Return immediately
    return JobStartResponse(
        job_id=job_id,
        status="pending",
        message=f"Weekly analysis started for config '{config_key}'",
        config_key=config_key,
        config_name=config.name,
    )


@app.get(
    "/schedules/today",
    response_model=TodayJobsResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def get_today_jobs_endpoint() -> TodayJobsResponse:
    """
    Get all jobs that ran today or are scheduled to run today.

    Returns:
        TodayJobsResponse with lists of realized and scheduled jobs

    """
    try:
        now = datetime.now(TZ)
        today = now.date()

        # 1. Get all jobs created today
        all_today_jobs = job_manager.get_jobs_by_date(today)

        # 2. Filter to realized jobs (exclude PENDING - only jobs that
        # have actually started)
        realized_jobs = [
            job
            for job in all_today_jobs
            if job.status != JobStatus.PENDING
        ]

        # 3. Get all schedules to find what else is planned for today
        all_schedules = get_all_schedules()
        scheduled_jobs: list[ScheduledJob] = []

        # Helper to check if a job has already been realized or created
        # today
        def has_started(config_key: str, job_type: JobType) -> bool:
            # Check if job is already realized
            # (i.e. started, completed, or failed)
            if any(
                j.config_key == config_key and j.type == job_type
                for j in realized_jobs
            ):
                return True
            # Also check if job has been created but is still pending
            return any(
                j.config_key == config_key
                and j.type == job_type
                and j.status == JobStatus.PENDING
                for j in all_today_jobs
            )

        for schedule in all_schedules:
            config_key = schedule["key"]
            config_name = schedule["name"]

            # Check daily scrape
            daily = schedule.get("daily_scrape", {})
            if daily.get("enabled") and not has_started(
                config_key, JobType.SCRAPE,
            ):
                scheduled_jobs.append(
                    ScheduledJob(
                        type=JobType.SCRAPE,
                        config_key=config_key,
                        config_name=config_name,
                        scheduled_at=f"{daily['hour']:02d}:{daily['minute']:02d}",
                    ),
                )

            # Check weekly analysis
            weekly = schedule.get("weekly_analysis", {})
            if weekly.get("enabled") and not has_started(
                config_key, JobType.ANALYSIS,
            ):
                # Check if today is the scheduled day
                day_map = {
                    "mon": 0,
                    "tue": 1,
                    "wed": 2,
                    "thu": 3,
                    "fri": 4,
                    "sat": 5,
                    "sun": 6,
                }
                scheduled_day = day_map.get(
                    weekly.get("day_of_week", "").lower(),
                )
                if scheduled_day == today.weekday():
                    scheduled_jobs.append(
                        ScheduledJob(
                            type=JobType.ANALYSIS,
                            config_key=config_key,
                            config_name=config_name,
                            scheduled_at=f"{weekly['hour']:02d}:{weekly['minute']:02d}",
                        ),
                    )

        return TodayJobsResponse(
            realized_jobs=realized_jobs,
            scheduled_jobs=scheduled_jobs,
        )

    except Exception as e:
        logger.exception("Error fetching today's jobs")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/jobs/{job_id}",
    response_model=Job,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
def get_job_status(job_id: str) -> Job:
    """
    Get the status of a background job.

    Args:
        job_id: The ID of the job to retrieve

    Returns:
        Job object with current status and results

    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found",
        )
    return job


@app.get(
    "/schedules",
    response_model=SchedulesResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def get_schedules_endpoint() -> SchedulesResponse:
    """
    Get all active config schedules.

    Returns schedules in cron format for cloud scheduler integration.

    Returns:
        SchedulesResponse with list of config schedules

    """
    try:
        schedules = get_all_schedules()
        return SchedulesResponse(
            configs=[ScheduleConfig(**schedule) for schedule in schedules],
        )

    except Exception as e:
        logger.exception("Error fetching schedules")
        raise HTTPException(status_code=500, detail=str(e)) from e


