"""Stage 5: publish a tested agent.

A deployment is currently a versioned copy of the agent file under
data/deployments/<build_id>/ plus a manifest. Each build gets its own
directory, so deployments never overwrite each other. Serving agents as
isolated APIs is planned for a later phase.

The deployer refuses code whose TestResult did not pass. The pipeline already
enforces this; the check here makes it impossible to bypass by calling the
deployer directly.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from metaagent.core.config import Settings, get_settings
from metaagent.schemas import AgentType, DeploymentResult, StageError, StageStatus, TestResult, utcnow

logger = logging.getLogger(__name__)


class DeploymentRefused(Exception):
    pass


class Stage5Deployer:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    @property
    def deploy_dir(self) -> Path:
        return self.settings.deployments_dir

    def deploy(self, build_id: str, code: str, agent_type: AgentType,
               test_result: Optional[TestResult]) -> DeploymentResult:
        started = time.perf_counter()
        if test_result is None or not test_result.passed:
            raise DeploymentRefused("Deployment refused: the agent did not pass Stage 4 testing")

        try:
            target = self.deploy_dir / build_id
            target.mkdir(parents=True, exist_ok=False)
            agent_file = target / "agent.py"
            agent_file.write_text(code, encoding="utf-8")

            # The RAG template ships a Streamlit upload UI as its entry point.
            launcher = "streamlit run" if "import streamlit" in code else "python"
            run_command = f'{launcher} "{agent_file}"'
            manifest = {
                "deployment_id": build_id,
                "agent_type": agent_type,
                "file": agent_file.name,
                "run_command": run_command,
                "deployed_at": utcnow().isoformat(),
                "tests": [c.model_dump() for c in test_result.checks],
            }
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except OSError as e:
            logger.error("Deployment failed: %s", e)
            return DeploymentResult(
                status=StageStatus.FAILED,
                error=StageError(type=type(e).__name__, message=f"Could not write deployment: {e}"),
                duration_s=time.perf_counter() - started,
            )

        return DeploymentResult(
            deployment_id=build_id, agent_type=agent_type,
            deployed_file=str(agent_file), run_command=run_command,
            duration_s=time.perf_counter() - started,
        )
