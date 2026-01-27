"""Data types for NewsBot views and services."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class ReportInfo:
    """Information about a report file."""

    filename: str
    modified: datetime
    size: int
    storage: Literal["local", "supabase"]


@dataclass
class LogFileInfo:
    """Information about a log file."""

    filename: str
    config_name: str
    modified: datetime
    size: int
    is_active: bool


@dataclass
class ConfigWithReports:
    """Configuration with associated report information."""

    name: str
    key: str
    report_count: int
    latest_report: str
    last_modified: datetime
    storage: Literal["local", "supabase"]
