"""Stage 3: produce agent source code.

The generator only *produces* code (one LLM attempt per call, or a template).
It never executes anything: testing and the self-correction loop are driven
by the pipeline, which feeds evaluator errors back in through `feedback`.
"""

import json
import logging
import time
from typing import Dict, List, Optional

from metaagent.ai import llm as llm_factory
from metaagent.ai.generator.prompts import CODEGEN_PROMPT, FEEDBACK_TEMPLATE
from metaagent.ai.generator.scorer import CodeQualityScorer
from metaagent.ai.generator.templates import render_template
from metaagent.core.config import Settings, get_settings
from metaagent.schemas import (
    CodeGenerationResult, Plan, StageError, StageStatus, ToolChoice, ToolSelectionResult,
)
from metaagent.tools.registry import get_implementation

logger = logging.getLogger(__name__)


def extract_code(raw) -> str:
    text = llm_factory.response_text(raw)
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def finalize(code: str, method: str, model: Optional[str], started: float) -> CodeGenerationResult:
    return CodeGenerationResult(
        code=code, method=method, model=model,
        lines=len(code.splitlines()),
        quality_score=CodeQualityScorer.score(code),
        duration_s=time.perf_counter() - started,
    )


class CodeGenerator:
    def __init__(self, llm=None, model: Optional[str] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model = llm_factory.resolve_model(model, self.settings)
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = llm_factory.chat_model(self.model, self.settings.codegen_temperature, self.settings)
        return self._llm

    def generate_llm(self, plan: Plan, selection: ToolSelectionResult, user_request: str,
                     previous_errors: Optional[List[str]] = None) -> CodeGenerationResult:
        """One LLM generation attempt. Returns a FAILED result instead of raising."""
        started = time.perf_counter()
        feedback = ""
        if previous_errors:
            feedback = FEEDBACK_TEMPLATE.format(
                errors="\n".join(f"- Attempt {i + 1}: {e}" for i, e in enumerate(previous_errors)))
        try:
            response = self.llm.invoke(CODEGEN_PROMPT.format(
                agent_type=plan.agent_type,
                selected_tools=json.dumps({t: c.implementation for t, c in selection.selections.items()}),
                tool_snippets=self._snippets(selection.selections) or "None",
                user_request=user_request,
                model=self.settings.agent_model,
                embedding_model=self.settings.embedding_model,
                error_feedback=feedback,
            ))
        except Exception as e:
            logger.warning("Code generation LLM call failed: %s", e)
            return self._failed("LLMError",
                                f"Code generation LLM call failed: {llm_factory.describe_error(e)}", started)

        code = extract_code(response)
        if "class Agent" not in code or "def run" not in code:
            return self._failed("InvalidOutput", "LLM output does not define class Agent with run()",
                                started, code=code or None)
        return finalize(code, "llm", self.model, started)

    def generate_template(self, plan: Plan, selection: ToolSelectionResult,
                          user_request: str) -> CodeGenerationResult:
        started = time.perf_counter()
        try:
            code = render_template(plan.agent_type, selection.selections, user_request,
                                   method="template", settings=self.settings)
        except Exception as e:
            logger.error("Template rendering failed: %s", e)
            return self._failed(type(e).__name__, f"Template rendering failed: {e}", started)
        return finalize(code, "template", None, started)

    def _snippets(self, selections: Dict[str, ToolChoice]) -> str:
        parts = []
        for tool, choice in selections.items():
            impl = get_implementation(tool, choice.implementation)
            if impl and impl.available:
                parts.append(f"# {tool}: {impl.id}\n{impl.render_snippet(self.settings).strip()}")
        return "\n\n".join(parts)

    def _failed(self, error_type: str, message: str, started: float,
                code: Optional[str] = None) -> CodeGenerationResult:
        return CodeGenerationResult(
            status=StageStatus.FAILED,
            error=StageError(type=error_type, message=message),
            code=code, method="llm", model=self.model,
            duration_s=time.perf_counter() - started,
        )
