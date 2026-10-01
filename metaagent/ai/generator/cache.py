"""Local cache of LLM-generated agent code that already passed testing.

Rules (enforced by the pipeline, which is the only writer):
  * only LLM output that passed Stage 4 is stored, never failures or templates
  * the key covers request, agent type, selected implementations, constraints,
    model, prompt version and template version, so changing any of them
    invalidates old entries
  * a hit is re-tested before use; entries that no longer pass are evicted
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional

from metaagent.ai.generator.prompts import PROMPT_VERSION
from metaagent.ai.generator.templates import TEMPLATE_VERSION
from metaagent.schemas import Constraints, utcnow

logger = logging.getLogger(__name__)

CACHE_FORMAT = 2


def make_key(user_request: str, agent_type: str, implementations: Dict[str, str],
             constraints: Constraints, model: str) -> str:
    payload = {
        "format": CACHE_FORMAT,
        "prompt": PROMPT_VERSION,
        "template": TEMPLATE_VERSION,
        "request": user_request.strip(),
        "agent_type": agent_type,
        "implementations": dict(sorted(implementations.items())),
        "constraints": constraints.model_dump(),
        "model": model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CodeCache:
    _lock = threading.Lock()

    def __init__(self, cache_file: Path):
        self.cache_file = Path(cache_file)

    def _read(self) -> dict:
        if not self.cache_file.exists():
            return {}
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Ignoring unreadable cache file %s: %s", self.cache_file, e)
            return {}

    def _write(self, data: dict) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a crash never leaves a half-written cache.
        fd, tmp = tempfile.mkstemp(dir=self.cache_file.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.cache_file)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._read().get(key)

    def put(self, key: str, code: str, model: str) -> None:
        with self._lock:
            data = self._read()
            data[key] = {"code": code, "model": model, "stored_at": utcnow().isoformat()}
            self._write(data)

    def evict(self, key: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(key, None) is not None:
                self._write(data)
