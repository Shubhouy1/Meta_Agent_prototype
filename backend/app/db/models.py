"""Database tables (SQLModel). Schema changes go through Alembic migrations
in backend/migrations; never edit a deployed schema by hand."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class BuildRecord(SQLModel, table=True):
    __tablename__ = "builds"

    id: str = Field(primary_key=True, max_length=64)
    owner: str = Field(index=True, max_length=100)
    status: str = Field(index=True, max_length=20)
    request: str
    model: Optional[str] = Field(default=None, max_length=100)
    agent_type: Optional[str] = Field(default=None, max_length=20)
    failed_stage: Optional[str] = Field(default=None, max_length=30)
    error_message: Optional[str] = None
    deployed: bool = Field(default=False, index=True)
    created_at: datetime = Field(index=True)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_s: Optional[float] = None
    # Full engine BuildResult, serialised. The columns above are denormalised
    # copies for filtering and listing without parsing JSON.
    result_json: str


class BuildEventRecord(SQLModel, table=True):
    __tablename__ = "build_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    build_id: str = Field(foreign_key="builds.id", index=True, max_length=64)
    kind: str = Field(max_length=10)            # "stage" (pipeline event) or "status" (lifecycle)
    stage: Optional[str] = Field(default=None, max_length=30)
    event: str = Field(max_length=30)
    message: str = ""
    attempt: Optional[int] = None
    created_at: datetime
