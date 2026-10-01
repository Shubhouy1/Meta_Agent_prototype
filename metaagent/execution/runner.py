"""The ONLY place where generated agent code is executed.

Each run is a separate Python subprocess with:
  * a temporary working directory outside the project tree
  * an allowlisted environment: no API keys, no tokens, no .env loading
  * a hard wall-clock timeout (the process is killed, not abandoned)
  * a cap on LLM/embedding calls, which are proxied through this process

Two modes:
  TEST  LLM calls get a canned reply, embeddings get deterministic vectors.
        No network, no cost, deterministic. Used by Stage 4.
  LIVE  LLM/embedding calls are made here, in the trusted parent, with the
        configured Gemini client. Used by the "Live agent test" in the UI.

What this does NOT do yet (planned for the deployment phase): restrict the
filesystem, restrict network access, or limit memory/CPU. Generated code can
still read any file the current user can read. Do not expose this runner to
untrusted users.
"""

import hashlib
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from metaagent.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

BOOTSTRAP = Path(__file__).with_name("_bootstrap.py")
TEST_LLM_REPLY = "This is a MetaAgent test response."
TEST_EMBEDDING_DIM = 16

# Environment variables a Python child genuinely needs. Everything else,
# including GOOGLE_API_KEY and any *_KEY/*_TOKEN/*_SECRET, is dropped.
_ENV_ALLOWLIST = (
    "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
    "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH",
    "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)


class RunMode(str, Enum):
    TEST = "test"
    LIVE = "live"


@dataclass
class RunResult:
    ok: bool
    output: Optional[str] = None
    output_type: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    timed_out: bool = False
    duration_s: float = 0.0
    run_time_s: Optional[float] = None
    llm_calls: int = 0
    # Proxied model calls that failed (quota, auth, limit). The agent may have
    # caught these and returned its own text, so callers should surface them.
    proxy_errors: List[str] = field(default_factory=list)
    stderr_tail: str = ""


def _fake_vector(text: str) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:TEST_EMBEDDING_DIM]]


def _child_env(extra: Optional[dict] = None) -> dict:
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ANONYMIZED_TELEMETRY": "False",
    })
    env.update(extra or {})
    return env


class AgentRunner:
    def __init__(self, settings: Optional[Settings] = None,
                 live_chat: Optional[Callable[[list, Optional[str]], str]] = None,
                 live_embed: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self.settings = settings or get_settings()
        self._live_chat = live_chat
        self._live_embed = live_embed

    # ── public API ──────────────────────────────────────

    def run_code(self, code: str, user_input: str, mode: RunMode = RunMode.TEST,
                 timeout_s: Optional[float] = None) -> RunResult:
        # ignore_cleanup_errors: on Windows a killed child can briefly keep files locked.
        with tempfile.TemporaryDirectory(prefix="metaagent_run_", ignore_cleanup_errors=True) as workdir:
            agent_path = Path(workdir) / "agent.py"
            agent_path.write_text(code, encoding="utf-8")
            return self._run(agent_path, Path(workdir), user_input, mode, timeout_s)

    def run_file(self, path: Path, user_input: str, mode: RunMode = RunMode.LIVE,
                 timeout_s: Optional[float] = None) -> RunResult:
        code = Path(path).read_text(encoding="utf-8")
        return self.run_code(code, user_input, mode, timeout_s)

    # ── internals ───────────────────────────────────────

    def _run(self, agent_path: Path, workdir: Path, user_input: str, mode: RunMode,
             timeout_s: Optional[float]) -> RunResult:
        if timeout_s is None:
            timeout_s = self.settings.test_run_timeout_s if mode == RunMode.TEST else self.settings.live_run_timeout_s
        chroma_dir = workdir / "chroma_db" if mode == RunMode.TEST else self.settings.chroma_dir

        started = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-B", str(BOOTSTRAP), str(agent_path)],
            cwd=str(workdir), env=_child_env({"CHROMA_DIR": str(chroma_dir)}),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        stderr_chunks: List[str] = []

        def read_stdout():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        def read_stderr():
            for line in proc.stderr:
                stderr_chunks.append(line)
                if len(stderr_chunks) > 200:
                    del stderr_chunks[:100]

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()

        llm_calls = 0
        proxy_errors: List[str] = []
        result: Optional[RunResult] = None
        try:
            self._send(proc, {"type": "run", "input": user_input})
            deadline = started + timeout_s
            while result is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    result = RunResult(ok=False, timed_out=True, error_type="Timeout",
                                       error=f"Agent did not finish within {timeout_s:.0f}s")
                    break
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    continue
                if line is None:
                    result = RunResult(ok=False, error_type="ProcessExited",
                                       error="Agent process exited without a result")
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = message.get("type")
                if kind == "result":
                    result = self._to_result(message)
                elif kind in ("llm", "embed"):
                    llm_calls += 1
                    if llm_calls > self.settings.max_llm_calls_per_run:
                        reply = {"error": f"LLM call limit ({self.settings.max_llm_calls_per_run}) exceeded"}
                    else:
                        reply = self._answer(message, mode)
                    if "error" in reply:
                        proxy_errors.append(reply["error"])
                    self._send(proc, reply)
        except (BrokenPipeError, OSError) as e:
            result = RunResult(ok=False, error_type="ProcessError", error=str(e))
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        result.duration_s = time.perf_counter() - started
        result.llm_calls = llm_calls
        result.proxy_errors = proxy_errors
        result.stderr_tail = "".join(stderr_chunks)[-3000:]
        logger.info("Agent run mode=%s ok=%s timed_out=%s llm_calls=%d duration=%.2fs",
                    mode.value, result.ok, result.timed_out, llm_calls, result.duration_s)
        return result

    @staticmethod
    def _send(proc: subprocess.Popen, message: dict) -> None:
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _to_result(message: dict) -> RunResult:
        if message.get("ok"):
            output = message.get("output")
            return RunResult(
                ok=True, output=output,
                output_type=message.get("output_type"),
                run_time_s=message.get("run_time_s"),
                error=None if output is not None else f"run() returned {message.get('output_type')}",
            )
        return RunResult(ok=False, error_type=message.get("error_type"),
                         error=message.get("error"), traceback=message.get("traceback"))

    def _answer(self, message: dict, mode: RunMode) -> dict:
        try:
            if message["type"] == "llm":
                if mode == RunMode.TEST:
                    return {"text": TEST_LLM_REPLY}
                return {"text": self._chat(message.get("messages", []), message.get("model"))}
            texts = message.get("texts", [])
            if mode == RunMode.TEST:
                return {"vectors": [_fake_vector(t) for t in texts]}
            return {"vectors": self._embed(texts)}
        except Exception as e:
            from metaagent.ai import llm as llm_factory

            logger.warning("Proxied %s call failed: %s", message.get("type"), e)
            # The child (and any text the agent returns) only sees the short summary.
            return {"error": llm_factory.describe_error(e)}

    def _chat(self, messages: list, requested_model: Optional[str]) -> str:
        if self._live_chat:
            return self._live_chat(messages, requested_model)
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from metaagent.ai import llm as llm_factory

        model = requested_model if requested_model in self.settings.available_models else self.settings.agent_model
        roles = {"system": SystemMessage, "ai": AIMessage}
        lc_messages = [roles.get(m.get("role"), HumanMessage)(content=m.get("content", "")) for m in messages]
        client = llm_factory.chat_model(model, 0.3, self.settings)
        return llm_factory.response_text(client.invoke(lc_messages))

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self._live_embed:
            return self._live_embed(texts)
        from metaagent.ai import llm as llm_factory

        return llm_factory.embeddings(self.settings).embed_documents(texts)
