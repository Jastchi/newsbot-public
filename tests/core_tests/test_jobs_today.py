"""
Tests for the /schedules/today endpoint in the API.
"""

from datetime import datetime

from newsbot.constants import TZ
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.job_manager import JobStatus, JobType, job_manager


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.app import app
    return TestClient(app)


@pytest.fixture
def clean_job_manager():
    """Clear jobs before each test."""
    job_manager._jobs = {}
    return job_manager


def test_jobs_today_endpoint_realized_and_scheduled(client, monkeypatch, clean_job_manager):
    """Test that /schedules/today returns both realized and scheduled jobs."""
    
    # Thursday, Jan 22, 2026
    fixed_now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=TZ)
    
    # Mock datetime.now() in both api.app and job_manager
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now
    
    monkeypatch.setattr("api.app.datetime", MockDatetime)
    monkeypatch.setattr("api.job_manager.datetime", MockDatetime)
    
    # Mock get_all_schedules (daily scrape at fixed 00:05)
    mock_schedules = [
        {
            "key": "config1",
            "name": "Config 1",
            "daily_scrape": {"enabled": True},
            "weekly_analysis": {
                "enabled": True,
                "day_of_week": "thu",  # Today
                "hour": 15,
                "minute": 0,
            },
        },
        {
            "key": "config2",
            "name": "Config 2",
            "daily_scrape": {"enabled": True},
            "weekly_analysis": {
                "enabled": True,
                "day_of_week": "mon",  # Not today
                "hour": 9,
                "minute": 0,
            },
        },
    ]
    monkeypatch.setattr("api.app.get_all_schedules", Mock(return_value=mock_schedules))
    
    # 1. Add a realized job for config1 scrape (already started)
    job_id = clean_job_manager.create_job(JobType.SCRAPE, "config1")
    clean_job_manager.update_job(job_id, JobStatus.COMPLETED)
    
    response = client.get("/schedules/today")
    assert response.status_code == 200
    data = response.json()
    
    # Realized jobs should have 1 job (config1 scrape)
    assert len(data["realized_jobs"]) == 1
    assert data["realized_jobs"][0]["config_key"] == "config1"
    assert data["realized_jobs"][0]["type"] == "scrape"
    
    # Scheduled jobs:
    # config1: scrape (already started, so skip), analysis (scheduled for today, mon=0...thu=3)
    # config2: scrape (scheduled daily), analysis (monday, so skip)
    assert len(data["scheduled_jobs"]) == 2
    
    # Verify scheduled jobs
    sched_keys = [s["config_key"] for s in data["scheduled_jobs"]]
    assert "config1" in sched_keys
    assert "config2" in sched_keys
    
    # Find config1 analysis
    c1_analysis = next(s for s in data["scheduled_jobs"] if s["config_key"] == "config1" and s["type"] == "analysis")
    assert c1_analysis["scheduled_at"] == "15:00"
    
    # Find config2 scrape (fixed 00:05)
    c2_scrape = next(s for s in data["scheduled_jobs"] if s["config_key"] == "config2" and s["type"] == "scrape")
    assert c2_scrape["scheduled_at"] == "00:05"


def test_jobs_today_empty(client, monkeypatch, clean_job_manager):
    """Test /schedules/today when no jobs or schedules exist."""
    fixed_now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=TZ)
    
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now
            
    monkeypatch.setattr("api.app.datetime", MockDatetime)
    monkeypatch.setattr("api.job_manager.datetime", MockDatetime)
    monkeypatch.setattr("api.app.get_all_schedules", Mock(return_value=[]))
    
    response = client.get("/schedules/today")
    assert response.status_code == 200
    data = response.json()
    
    assert data["realized_jobs"] == []
    assert data["scheduled_jobs"] == []


def test_pending_jobs_excluded_from_realized(client, monkeypatch, clean_job_manager):
    """Test that PENDING jobs are excluded from realized_jobs."""
    fixed_now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=TZ)
    
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now
    
    monkeypatch.setattr("api.app.datetime", MockDatetime)
    monkeypatch.setattr("api.job_manager.datetime", MockDatetime)
    monkeypatch.setattr("api.app.get_all_schedules", Mock(return_value=[]))
    
    # Create jobs with different statuses
    pending_job_id = clean_job_manager.create_job(JobType.SCRAPE, "config1")
    # PENDING is the default status, so no need to update
    
    running_job_id = clean_job_manager.create_job(JobType.ANALYSIS, "config1")
    clean_job_manager.update_job(running_job_id, JobStatus.RUNNING)
    
    completed_job_id = clean_job_manager.create_job(JobType.SCRAPE, "config2")
    clean_job_manager.update_job(completed_job_id, JobStatus.COMPLETED)
    
    failed_job_id = clean_job_manager.create_job(JobType.ANALYSIS, "config2")
    clean_job_manager.update_job(failed_job_id, JobStatus.FAILED)
    
    response = client.get("/schedules/today")
    assert response.status_code == 200
    data = response.json()
    
    # Realized jobs should only include RUNNING, COMPLETED, and FAILED
    # PENDING should be excluded
    assert len(data["realized_jobs"]) == 3
    
    realized_statuses = [job["status"] for job in data["realized_jobs"]]
    assert JobStatus.PENDING.value not in realized_statuses
    assert JobStatus.RUNNING.value in realized_statuses
    assert JobStatus.COMPLETED.value in realized_statuses
    assert JobStatus.FAILED.value in realized_statuses


def test_pending_jobs_excluded_from_scheduled(client, monkeypatch, clean_job_manager):
    """Test that PENDING jobs are excluded from scheduled_jobs."""
    fixed_now = datetime(2026, 1, 22, 12, 0, 0, tzinfo=TZ)
    
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now
    
    monkeypatch.setattr("api.app.datetime", MockDatetime)
    monkeypatch.setattr("api.job_manager.datetime", MockDatetime)
    
    # Mock schedules (daily scrape at fixed 00:05)
    mock_schedules = [
        {
            "key": "config1",
            "name": "Config 1",
            "daily_scrape": {"enabled": True},
            "weekly_analysis": {"enabled": False},
        },
    ]
    monkeypatch.setattr("api.app.get_all_schedules", Mock(return_value=mock_schedules))
    
    # Create a PENDING job for config1 scrape
    pending_job_id = clean_job_manager.create_job(JobType.SCRAPE, "config1")
    # PENDING is the default status
    
    response = client.get("/schedules/today")
    assert response.status_code == 200
    data = response.json()
    
    # Scheduled jobs should NOT include config1 scrape because it's already
    # been created (even though it's PENDING)
    assert len(data["scheduled_jobs"]) == 0
    
    # But if we complete the job, it should be in realized_jobs
    clean_job_manager.update_job(pending_job_id, JobStatus.COMPLETED)
    
    response = client.get("/schedules/today")
    assert response.status_code == 200
    data = response.json()
    
    # Now it should be in realized_jobs
    assert len(data["realized_jobs"]) == 1
    assert data["realized_jobs"][0]["config_key"] == "config1"
    assert data["realized_jobs"][0]["status"] == JobStatus.COMPLETED.value
    # Still not in scheduled_jobs because it's realized
    assert len(data["scheduled_jobs"]) == 0
