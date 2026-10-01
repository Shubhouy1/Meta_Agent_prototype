"""Application service for builds: what a UI, CLI or API calls.

    service = BuildService()
    result = service.start_build(request, constraints, model, on_event=...)

It runs the engine pipeline, persists the result through a BuildStore, and
runs deployed agents for live invocation. Callers never touch the planner,
generator or runner directly.
"""

import logging
from pathlib import Path
from typing import Optional

from metaagent.builds.store import BuildStats, BuildStore, JsonBuildStore
from metaagent.core.config import Settings, get_settings
from metaagent.core.logging import configure_logging
from metaagent.execution.runner import AgentRunner, RunMode, RunResult
from metaagent.pipeline import EventCallback, run_build
from metaagent.schemas import BuildResult, Constraints

logger = logging.getLogger(__name__)


class BuildService:
    def __init__(self, settings: Optional[Settings] = None, store: Optional[BuildStore] = None,
                 runner: Optional[AgentRunner] = None, llm=None):
        """`llm` overrides the Gemini client for every stage (tests, alternative callers)."""
        configure_logging()
        self.settings = settings or get_settings()
        self.store = store or JsonBuildStore(self.settings.builds_dir)
        self.runner = runner or AgentRunner(self.settings)
        self._llm = llm

    def start_build(self, request: str, constraints: Optional[Constraints] = None,
                    model: Optional[str] = None, on_event: Optional[EventCallback] = None,
                    use_cache: bool = True, build_id: Optional[str] = None) -> BuildResult:
        result = run_build(request, constraints, model, build_id=build_id, on_event=on_event,
                           llm=self._llm, runner=self.runner, settings=self.settings, use_cache=use_cache)
        try:
            self.store.save(result)
        except Exception as e:
            logger.error("Could not save build record %s: %s", result.build_id, e)
        return result

    def run_deployed_agent(self, result: BuildResult, user_input: str) -> RunResult:
        """Run a deployed agent once with real model access (through the runner proxy)."""
        if not result.deployed:
            raise ValueError("This build was not deployed")
        path = Path(result.deployment.deployed_file)
        if not path.is_file():
            raise FileNotFoundError(f"Deployed file not found: {path}")
        return self.runner.run_file(path, user_input, mode=RunMode.LIVE)

    def stats(self) -> BuildStats:
        return self.store.stats()
