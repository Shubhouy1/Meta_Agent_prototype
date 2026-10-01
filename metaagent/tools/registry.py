"""The single catalog of tools MetaAgent can put into a generated agent.

Each implementation carries both the metadata Stage 2 scores and the code
snippet Stage 3 embeds, so the two can no longer drift apart. Implementations
without a working snippet are listed (they appear in the what-if comparison)
but marked unavailable, and the selector never picks them.

Snippets use string.Template placeholders ($embedding_model); everything else
is literal Python, so braces need no escaping.
"""

from dataclasses import dataclass, field
from string import Template
from typing import Dict, List, Optional

from metaagent.core.config import Settings, get_settings


@dataclass(frozen=True)
class ToolImplementation:
    id: str
    name: str
    description: str
    cost: str
    api_key_required: bool
    performance: str
    setup_time: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    best_for: List[str] = field(default_factory=list)
    # Name of the function/class the snippet defines.
    entrypoint: Optional[str] = None
    # "function": callable(str) -> str.  "memory": class with store()/get_context().
    kind: str = "function"
    snippet: Optional[str] = None
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.snippet is not None and self.entrypoint is not None

    def render_snippet(self, settings: Optional[Settings] = None) -> str:
        if not self.available:
            raise ValueError(f"Implementation '{self.id}' has no snippet: {self.unavailable_reason}")
        settings = settings or get_settings()
        return Template(self.snippet).substitute(embedding_model=repr(settings.embedding_model))


_DUCKDUCKGO = '''
def search_tool(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return f"Search unavailable (install duckduckgo-search): {query}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No results: {query}"
        return "\\n".join(f"{i + 1}. {r['body'][:200]}" for i, r in enumerate(results))
    except Exception as e:
        return f"Search failed for {query}: {e}"
'''

_TAVILY = '''
def search_tool(query: str) -> str:
    import os
    try:
        from tavily import TavilyClient
    except ImportError:
        return f"Search unavailable (install tavily-python): {query}"
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query, max_results=3)
        return "\\n".join(r["content"][:200] for r in results["results"])
    except Exception as e:
        return f"Tavily search failed for {query}: {e}"
'''

_SAFE_EVAL = '''
def calculate_tool(expr: str) -> str:
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    unary = {ast.USub: op.neg, ast.UAdd: op.pos}

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    clean = expr.lower().replace("calculate", "").replace("what is", "").strip(" ?")
    try:
        return f"{clean} = {_eval(ast.parse(clean, mode='eval').body)}"
    except Exception:
        return f"Calculation error: {expr}"
'''

_CHROMADB = '''
def retriever_tool(query: str) -> str:
    import os
    try:
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(model=$embedding_model)
        db = Chroma(persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"), embedding_function=embeddings)
        docs = db.similarity_search(query, k=3)
        if not docs:
            return f"No documents found for: {query}"
        return "\\n".join(d.page_content[:300] for d in docs)
    except Exception as e:
        return f"Retrieval failed for {query}: {e}"
'''

_BUFFER_MEMORY = '''
class MemoryTool:
    def __init__(self):
        self.history = []

    def store(self, user_input, response):
        self.history.append(("user", user_input))
        self.history.append(("assistant", response))

    def get_context(self, last_n=4):
        recent = self.history[-last_n * 2:]
        return "\\n".join(f"{role}: {text}" for role, text in recent)
'''


TOOL_REGISTRY: Dict[str, List[ToolImplementation]] = {
    "search": [
        ToolImplementation(
            id="duckduckgo", name="DuckDuckGo Search", description="Free, no API key required",
            cost="Free", api_key_required=False, performance="Medium", setup_time="2 minutes",
            pros=["Free", "No API key", "Privacy focused"], cons=["Rate limited"],
            best_for=["Prototyping", "Hackathons"],
            entrypoint="search_tool", snippet=_DUCKDUCKGO,
        ),
        ToolImplementation(
            id="tavily", name="Tavily AI Search", description="LLM-optimized search API",
            cost="Paid (free tier available)", api_key_required=True, performance="High",
            setup_time="5 minutes", pros=["High quality", "LLM-optimized"],
            cons=["Paid after free tier"], best_for=["Production", "Research"],
            entrypoint="search_tool", snippet=_TAVILY,
        ),
    ],
    "calculator": [
        ToolImplementation(
            id="safe_eval", name="Python Safe Eval", description="AST-based safe evaluation",
            cost="Free", api_key_required=False, performance="High", setup_time="0 minutes",
            pros=["Safe", "No dependencies"], cons=["Basic math only"],
            best_for=["Basic math", "Educational"],
            entrypoint="calculate_tool", snippet=_SAFE_EVAL,
        ),
        ToolImplementation(
            id="numexpr", name="NumExpr", description="Fast numerical evaluation",
            cost="Free", api_key_required=False, performance="Very High", setup_time="2 minutes",
            pros=["Very fast", "Efficient"], cons=["External dependency"],
            best_for=["Performance-critical"],
            unavailable_reason="No snippet implemented yet",
        ),
    ],
    "retriever": [
        ToolImplementation(
            id="chromadb", name="ChromaDB", description="Lightweight vector database",
            cost="Free", api_key_required=False, performance="Medium", setup_time="3 minutes",
            pros=["Easy setup", "Persistent"], cons=["Memory intensive"],
            best_for=["Prototyping", "Small-medium datasets"],
            entrypoint="retriever_tool", snippet=_CHROMADB,
        ),
        ToolImplementation(
            id="faiss", name="FAISS", description="Facebook AI Similarity Search",
            cost="Free", api_key_required=False, performance="Very High", setup_time="5 minutes",
            pros=["Very fast", "Memory efficient"], cons=["No built-in persistence"],
            best_for=["High performance", "Large datasets"],
            unavailable_reason="No ingestion pipeline builds a FAISS index yet",
        ),
        ToolImplementation(
            id="pinecone", name="Pinecone", description="Managed vector database",
            cost="Paid", api_key_required=True, performance="High", setup_time="10 minutes",
            pros=["Scalable", "Cloud-native"], cons=["Paid", "API key required"],
            best_for=["Production", "Large scale"],
            unavailable_reason="No snippet implemented yet",
        ),
    ],
    "memory": [
        ToolImplementation(
            id="buffer_memory", name="Buffer Memory", description="Simple in-memory storage",
            cost="Free", api_key_required=False, performance="High", setup_time="0 minutes",
            pros=["Simple", "No setup"], cons=["No persistence"],
            best_for=["Prototyping", "Simple chatbots"],
            entrypoint="MemoryTool", kind="memory", snippet=_BUFFER_MEMORY,
        ),
        ToolImplementation(
            id="redis_memory", name="Redis Memory", description="Persistent Redis-backed memory",
            cost="Free (self-hosted)", api_key_required=False, performance="High",
            setup_time="15 minutes", pros=["Persistent", "Scalable"],
            cons=["Requires Redis server"], best_for=["Production", "Multi-user"],
            unavailable_reason="No snippet implemented yet",
        ),
    ],
}

# Used when the LLM selector fails and no constraint-based recommendation exists.
DEFAULT_IMPLEMENTATIONS = {
    "search": "duckduckgo",
    "calculator": "safe_eval",
    "retriever": "chromadb",
    "memory": "buffer_memory",
}


def get_implementation(tool: str, impl_id: str) -> Optional[ToolImplementation]:
    for impl in TOOL_REGISTRY.get(tool, []):
        if impl.id == impl_id:
            return impl
    return None


def available_implementations(tool: str) -> List[ToolImplementation]:
    return [impl for impl in TOOL_REGISTRY.get(tool, []) if impl.available]
