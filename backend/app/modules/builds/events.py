"""Build events: persisted, then fanned out to live subscribers.

The engine reports progress through the pipeline's existing `on_event`
callback (metaagent.schemas.BuildEvent). `EventBroker.publish_pipeline_event`
is that callback. Every event is first written to the build_events table
(so streams can replay history and resume with Last-Event-ID), then pushed
to subscribers' asyncio queues. Publishing happens on build worker threads;
delivery is handed to each subscriber's event loop thread-safely.

Two kinds of event:
  stage   pipeline progress (planning started, attempt_failed, ...)
  status  build lifecycle (queued, running, succeeded, failed); a succeeded
          or failed status event is always the last event of a build
"""

import asyncio
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from backend.app.core.errors import redact
from backend.app.db.build_store import SqlEventStore
from backend.app.db.models import BuildEventRecord
from metaagent.schemas import TERMINAL_STATUSES, BuildEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamEvent:
    id: int
    build_id: str
    kind: str
    event: str
    stage: Optional[str]
    message: str
    attempt: Optional[int]
    created_at: datetime

    @property
    def terminal(self) -> bool:
        return self.kind == "status" and self.event in TERMINAL_STATUSES

    @classmethod
    def from_record(cls, r: BuildEventRecord) -> "StreamEvent":
        created = r.created_at if r.created_at.tzinfo else r.created_at.replace(tzinfo=timezone.utc)
        return cls(id=r.id, build_id=r.build_id, kind=r.kind, event=r.event, stage=r.stage,
                   message=r.message, attempt=r.attempt, created_at=created)


class Subscription:
    def __init__(self, build_id: str):
        self.build_id = build_id
        self.loop = asyncio.get_running_loop()
        self.queue: "asyncio.Queue[StreamEvent]" = asyncio.Queue()


class EventBroker:
    def __init__(self, store: SqlEventStore):
        self._store = store
        self._subscribers: Dict[str, Set[Subscription]] = defaultdict(set)
        self._lock = threading.Lock()

    def publish(self, build_id: str, kind: str, event: str, stage: Optional[str] = None,
                message: str = "", attempt: Optional[int] = None) -> StreamEvent:
        record = self._store.append(build_id, kind, event, stage=stage, message=redact(message) or "",
                                    attempt=attempt)
        item = StreamEvent.from_record(record)
        with self._lock:
            subscribers = list(self._subscribers.get(build_id, ()))
        for sub in subscribers:
            try:
                sub.loop.call_soon_threadsafe(sub.queue.put_nowait, item)
            except RuntimeError:  # subscriber's loop already closed
                self.unsubscribe(sub)
        return item

    def publish_pipeline_event(self, event: BuildEvent) -> None:
        """The pipeline's on_event callback."""
        self.publish(event.build_id, "stage", event.event, stage=event.stage,
                     message=event.message, attempt=event.attempt)

    def history(self, build_id: str, after_id: int = 0) -> List[StreamEvent]:
        return [StreamEvent.from_record(r) for r in self._store.list_after(build_id, after_id)]

    def subscribe(self, build_id: str) -> Subscription:
        """Must be called from the event loop that will consume the queue."""
        sub = Subscription(build_id)
        with self._lock:
            self._subscribers[build_id].add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            subs = self._subscribers.get(sub.build_id)
            if subs is not None:
                subs.discard(sub)
                if not subs:
                    del self._subscribers[sub.build_id]
