"""Render the deterministic fallback agents from templates/*.py.tmpl.

Templates use string.Template ($name placeholders), so the Python code inside
them is written normally, with no brace escaping. Every value from the user
is inserted as a Python literal via repr(), so a request can't break out of
the generated code.
"""

from datetime import datetime
from pathlib import Path
from string import Template
from typing import Dict, Optional

from metaagent.core.config import Settings, get_settings
from metaagent.schemas import ToolChoice
from metaagent.tools.registry import get_implementation

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_VERSION = "2"

_TEMPLATE_FILES = {
    "chatbot": "chatbot.py.tmpl",
    "rag": "rag.py.tmpl",
    "tool_agent": "tool_agent.py.tmpl",
}


def _load(agent_type: str) -> Template:
    name = _TEMPLATE_FILES.get(agent_type, _TEMPLATE_FILES["chatbot"])
    return Template((TEMPLATE_DIR / name).read_text(encoding="utf-8"))


def _tool_parts(selections: Dict[str, ToolChoice], settings: Settings):
    code_parts, entries, memory_init = [], [], "None"
    for tool, choice in selections.items():
        impl = get_implementation(tool, choice.implementation)
        if impl is None or not impl.available:
            raise ValueError(f"Selected implementation '{choice.implementation}' for {tool} is not available")
        code_parts.append(impl.render_snippet(settings).strip())
        if impl.kind == "memory":
            memory_init = f"{impl.entrypoint}()"
        else:
            entries.append(f"            {tool!r}: {impl.entrypoint},")
    return "\n\n\n".join(code_parts), "\n".join(entries), memory_init


def render_template(agent_type: str, selections: Dict[str, ToolChoice], request: str,
                    method: str = "template", model: Optional[str] = None,
                    settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    tool_code, tool_entries, memory_init = _tool_parts(selections, settings)
    return _load(agent_type).substitute(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        method=method,
        request_literal=repr(request),
        model_literal=repr(model or settings.agent_model),
        embedding_model_literal=repr(settings.embedding_model),
        tool_code=tool_code or "# No tools selected",
        tool_entries=tool_entries,
        memory_init=memory_init,
    )
