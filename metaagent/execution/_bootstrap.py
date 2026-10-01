"""Child-process entry point used by execution/runner.py. Do not import this module.

Protocol: one JSON object per line. The child's real stdout carries protocol
messages only; anything the agent prints is redirected to stderr. The
parent sends {"type": "run", "input": ...}; the child answers LLM/embedding
requests through the parent and finishes with {"type": "result", ...}.

The child never receives API keys. langchain_google_genai's chat and
embedding classes are replaced with proxies that ask the parent to perform
the call, so the parent decides which model runs and how often.
"""

import importlib.util
import io
import json
import sys
import time
import traceback

_proto_out = sys.stdout
_proto_in = sys.stdin
sys.stdout = sys.stderr          # agent prints must not corrupt the protocol
sys.stdin = io.StringIO("")      # input() raises EOFError instead of reading the protocol


def _send(message: dict) -> None:
    _proto_out.write(json.dumps(message) + "\n")
    _proto_out.flush()


def _request(message: dict) -> dict:
    _send(message)
    line = _proto_in.readline()
    if not line:
        raise RuntimeError("MetaAgent runner closed the connection")
    reply = json.loads(line)
    if reply.get("error"):
        raise RuntimeError(reply["error"])
    return reply


def _disable_dotenv() -> None:
    try:
        import dotenv
    except ImportError:
        return
    dotenv.load_dotenv = lambda *args, **kwargs: False
    dotenv.find_dotenv = lambda *args, **kwargs: ""
    dotenv.dotenv_values = lambda *args, **kwargs: {}


def _install_llm_proxy() -> None:
    import langchain_google_genai as genai_pkg
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import ConfigDict

    def _content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content)

    class ProxyChatModel(BaseChatModel):
        model_config = ConfigDict(extra="allow")

        @property
        def _llm_type(self) -> str:
            return "metaagent-proxy"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            reply = _request({
                "type": "llm",
                "model": getattr(self, "model", None),
                "temperature": getattr(self, "temperature", None),
                "messages": [{"role": m.type, "content": _content(m.content)} for m in messages],
            })
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply["text"]))])

    class ProxyEmbeddings(Embeddings):
        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, texts):
            return _request({"type": "embed", "texts": list(texts)})["vectors"]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    genai_pkg.ChatGoogleGenerativeAI = ProxyChatModel
    genai_pkg.GoogleGenerativeAIEmbeddings = ProxyEmbeddings
    for submodule, attr, proxy in (("chat_models", "ChatGoogleGenerativeAI", ProxyChatModel),
                                   ("embeddings", "GoogleGenerativeAIEmbeddings", ProxyEmbeddings)):
        module = sys.modules.get(f"langchain_google_genai.{submodule}")
        if module is not None:
            setattr(module, attr, proxy)


def main() -> None:
    agent_path = sys.argv[1]
    run_request = json.loads(_proto_in.readline())
    try:
        with open(agent_path, encoding="utf-8") as f:
            source = f.read()
        _disable_dotenv()
        if "langchain_google_genai" in source:
            _install_llm_proxy()

        spec = importlib.util.spec_from_file_location("generated_agent", agent_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        agent_class = getattr(module, "Agent", None)
        if agent_class is None:
            raise AttributeError("Generated module does not define class Agent")

        agent = agent_class()
        started = time.perf_counter()
        output = agent.run(run_request["input"])
        _send({
            "type": "result", "ok": True,
            "output": output if isinstance(output, str) else None,
            "output_type": type(output).__name__,
            "output_repr": None if isinstance(output, str) else repr(output)[:500],
            "run_time_s": time.perf_counter() - started,
        })
    except BaseException as e:  # includes SystemExit from generated code
        _send({
            "type": "result", "ok": False,
            "error_type": type(e).__name__,
            "error": str(e)[:2000],
            "traceback": traceback.format_exc()[-4000:],
        })


if __name__ == "__main__":
    main()
