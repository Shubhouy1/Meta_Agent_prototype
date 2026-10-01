"""Build history persistence.

`BuildStore` is the interface the engine depends on. Implementations:
  * JsonBuildStore (here): one JSON file per build under data/builds/. Used by
    the Streamlit app and the CLI; no database required.
  * SqlBuildStore (backend/app/db): SQLite via SQLModel, used by the API.

Stores only read and write records; they contain no business logic.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

from metaagent.schemas import BuildResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildStats:
    total: int
    succeeded: int
    failed: int
    from_cache: int
    self_corrected: int


@runtime_checkable
class BuildStore(Protocol):
    def save(self, result: BuildResult) -> None: ...

    def get(self, build_id: str) -> Optional[BuildResult]: ...

    def list(self, limit: int = 50) -> List[BuildResult]: ...

    def stats(self) -> BuildStats: ...


def compute_stats(builds: List[BuildResult]) -> BuildStats:
    return BuildStats(
        total=len(builds),
        succeeded=sum(1 for b in builds if b.status == "succeeded"),
        failed=sum(1 for b in builds if b.status == "failed"),
        from_cache=sum(1 for b in builds if b.generation and b.generation.method == "cache"),
        self_corrected=sum(1 for b in builds if b.generation and b.generation.corrections > 0),
    )


class JsonBuildStore:
    def __init__(self, builds_dir: Path):
        self.builds_dir = Path(builds_dir)

    def save(self, result: BuildResult) -> None:
        self.builds_dir.mkdir(parents=True, exist_ok=True)
        path = self.builds_dir / f"{result.build_id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def get(self, build_id: str) -> Optional[BuildResult]:
        path = self.builds_dir / f"{build_id}.json"
        if not path.exists():
            return None
        return BuildResult.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 50) -> List[BuildResult]:
        if not self.builds_dir.exists():
            return []
        results = []
        for path in sorted(self.builds_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                results.append(BuildResult.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as e:  # one corrupt record must not hide the rest
                logger.warning("Skipping unreadable build record %s: %s", path.name, e)
        return results

    def stats(self) -> BuildStats:
        return compute_stats(self.list(limit=10_000))
