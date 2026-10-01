"""SQLite implementations of build persistence.

SqlBuildStore implements the engine's metaagent.builds.store.BuildStore
interface (save/get/list/stats), so the engine BuildService can use it
unchanged. The extra methods (create, mark_running, list_records,
fail_interrupted) cover the API's asynchronous lifecycle.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from backend.app.db.models import BuildEventRecord, BuildRecord
from metaagent.builds.store import BuildStats
from metaagent.schemas import BuildResult, StageError, utcnow

ACTIVE_STATUSES = ("queued", "running")


def _apply(record: BuildRecord, result: BuildResult) -> None:
    record.status = result.status
    record.request = result.request
    record.model = result.model
    record.agent_type = result.plan.plan.agent_type if result.plan and result.plan.ok else None
    record.failed_stage = result.failed_stage
    record.error_message = result.error.message if result.error else None
    record.deployed = result.deployed
    record.result_json = result.model_dump_json()
    if result.finished:
        record.finished_at = record.finished_at or utcnow()
        record.duration_s = result.duration_s


class SqlBuildStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    # ── engine BuildStore interface ──

    def save(self, result: BuildResult) -> None:
        with Session(self.engine) as session:
            record = session.get(BuildRecord, result.build_id)
            if record is None:  # builds started outside the API (e.g. tests, CLI)
                record = BuildRecord(id=result.build_id, owner="-", status=result.status,
                                     request=result.request, created_at=utcnow(), result_json="{}")
            _apply(record, result)
            session.add(record)
            session.commit()

    def get(self, build_id: str) -> Optional[BuildResult]:
        record = self.get_record(build_id)
        return BuildResult.model_validate_json(record.result_json) if record else None

    def list(self, limit: int = 50) -> List[BuildResult]:
        return [BuildResult.model_validate_json(r.result_json) for r in self.list_records(limit=limit)]

    def stats(self) -> BuildStats:
        with Session(self.engine) as session:
            rows = dict(session.exec(select(BuildRecord.status, func.count()).group_by(BuildRecord.status)).all())
            results = [BuildResult.model_validate_json(j) for j in session.exec(
                select(BuildRecord.result_json).where(BuildRecord.status.in_(("succeeded", "failed")))).all()]
        return BuildStats(
            total=sum(rows.values()),
            succeeded=rows.get("succeeded", 0),
            failed=rows.get("failed", 0),
            from_cache=sum(1 for b in results if b.generation and b.generation.method == "cache"),
            self_corrected=sum(1 for b in results if b.generation and b.generation.corrections > 0),
        )

    # ── API lifecycle ──

    def create(self, result: BuildResult, owner: str) -> BuildRecord:
        record = BuildRecord(id=result.build_id, owner=owner, status=result.status, request=result.request,
                             created_at=utcnow(), result_json="{}")
        _apply(record, result)
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def mark_running(self, build_id: str) -> None:
        with Session(self.engine) as session:
            record = session.get(BuildRecord, build_id)
            if record is None:
                return
            result = BuildResult.model_validate_json(record.result_json)
            result.status = "running"
            _apply(record, result)
            record.started_at = utcnow()
            session.add(record)
            session.commit()

    def get_record(self, build_id: str) -> Optional[BuildRecord]:
        with Session(self.engine) as session:
            return session.get(BuildRecord, build_id)

    def list_records(self, limit: int = 20, offset: int = 0, status: Optional[str] = None,
                     deployed: Optional[bool] = None) -> List[BuildRecord]:
        query = select(BuildRecord)
        if status:
            query = query.where(BuildRecord.status == status)
        if deployed is not None:
            query = query.where(BuildRecord.deployed == deployed)
        query = query.order_by(BuildRecord.created_at.desc()).offset(offset).limit(limit)
        with Session(self.engine) as session:
            return list(session.exec(query).all())

    def count_active(self) -> int:
        with Session(self.engine) as session:
            return session.exec(select(func.count()).select_from(BuildRecord)
                                .where(BuildRecord.status.in_(ACTIVE_STATUSES))).one()

    def fail_interrupted(self) -> List[str]:
        """Mark builds left queued/running by a previous process as failed."""
        failed = []
        with Session(self.engine) as session:
            for record in session.exec(select(BuildRecord).where(BuildRecord.status.in_(ACTIVE_STATUSES))).all():
                result = BuildResult.model_validate_json(record.result_json)
                result.status = "failed"
                result.error = StageError(type="Interrupted",
                                          message="The server stopped before this build finished.")
                _apply(record, result)
                session.add(record)
                failed.append(record.id)
            session.commit()
        return failed


class SqlEventStore:
    """Append-only log of build events; the autoincrement id is the SSE event id."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, build_id: str, kind: str, event: str, stage: Optional[str] = None,
               message: str = "", attempt: Optional[int] = None,
               created_at: Optional[datetime] = None) -> BuildEventRecord:
        record = BuildEventRecord(build_id=build_id, kind=kind, stage=stage, event=event,
                                  message=message, attempt=attempt, created_at=created_at or utcnow())
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def list_after(self, build_id: str, after_id: int = 0) -> List[BuildEventRecord]:
        query = (select(BuildEventRecord)
                 .where(BuildEventRecord.build_id == build_id, BuildEventRecord.id > after_id)
                 .order_by(BuildEventRecord.id))
        with Session(self.engine) as session:
            return list(session.exec(query).all())
