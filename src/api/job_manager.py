"""
Job Manager for tracking background API tasks.

Provides a simple in-memory job store and status tracking.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from newsbot.constants import TZ


class JobStatus(StrEnum):
    """Status of a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobType(StrEnum):
    """Type of a background job."""

    SCRAPE = "scrape"
    ANALYSIS = "analysis"


class Job(BaseModel):
    """Model representing a background job."""

    id: str
    type: JobType
    config_key: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(TZ))
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Manages background jobs in memory."""

    def __init__(self) -> None:
        """Initialize empty job store."""
        self._jobs: dict[str, Job] = {}

    def create_job(self, job_type: JobType, config_key: str) -> str:
        """
        Create a new pending job.

        Args:
            job_type: Type of job (scrape, analysis)
            config_key: Configuration key

        Returns:
            Job ID

        """
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, type=job_type, config_key=config_key)
        self._jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """
        Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job object or None if not found

        """
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Update job status and details.

        Args:
            job_id: Job ID
            status: New status
            result: Result dictionary (optional)
            error: Error message (optional)

        """
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.status = status
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now(TZ)
            if result:
                job.result = result
            if error:
                job.error = error

    def get_jobs_by_date(self, date: date) -> list[Job]:
        """
        Get all jobs created on a specific date.

        Args:
            date: The date to filter by

        Returns:
            List of Job objects

        """
        return [
            job
            for job in self._jobs.values()
            if job.created_at.date() == date
        ]


# Global job manager instance
job_manager = JobManager()
