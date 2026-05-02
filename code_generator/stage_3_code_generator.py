import os
import json
import time
import logging
import importlib.util
from typing import Dict, Optional, Tuple
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from .cache import CodeCache
from .sandbox import AdvancedSandbox
from .scorer import CodeQualityScorer
from .patterns import PatternDatabase
from .detector import ToolDetector

logger = logging.getLogger(__name__)


# ============================================
# TOOL IMPLEMENTATIONS (Embedded - Using Google Embeddings)
# ============================================

TOOL_IMPLEMENTATIONS = {
    "search": {
        "duckduckgo": '''
def search_tool(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return "\\n".join([f"{i+1}. {r['body'][:200]}" for i, r in enumerate(results)]) if results else f"No results: {query}"
    except ImportError:
        return f"Search for: {query} (install duckduckgo-search)"
''',
        "tavily": '''
def search_tool(query: str) -> str:
    import os
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query, max_results=3)
        return "\\n".join([r['content'][:200] for r in results['results']])
    except: 
        return f"Tavily search failed for {query}"
'''
    },
    "calculator": {
        "safe_eval": '''
def calculate_tool(expr: str) -> str:
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    def _eval(node):
        if isinstance(node, ast.Constant): 
            return node.value
        if isinstance(node, ast.BinOp): 
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise TypeError
    try:
        clean = expr.replace('calculate', '').strip()
        tree = ast.parse(clean, mode='eval')
        result = _eval(tree.body)
        return f"{clean} = {result}"
    except: 
        return f"Calculation error: {expr}"
'''
    },
    "retriever": {
        "chromadb": '''
def retriever_tool(query: str) -> str:
    try:
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        docs = db.similarity_search(query, k=3)
        return "\\n".join([d.page_content[:300] for d in docs])
    except Exception as e:
        return f"Retrieval failed for: {query}"
''',
        "faiss": '''
def retriever_tool(query: str) -> str:
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=3)
        return "\\n".join([d.page_content[:300] for d in docs])
    except Exception as e:
        return f"FAISS retrieval failed for: {query}"
'''
    },
    "memory": {
        "buffer": '''
class MemoryTool:
    def __init__(self):
        self.history = []
    def store(self, user_input, response):
        self.history.append(("user", user_input))
        self.history.append(("assistant", response))
    def get_context(self, last_n=4):
        recent = self.history[-last_n*2:] if self.history else []
        return "\\n".join([f"{r[0]}: {r[1]}" for r in recent])
'''
    }
}


# ============================================
# SELF-CORRECTING LLM PROMPT
# ============================================

SELF_CORRECTING_PROMPT = PromptTemplate(
    input_variables=["agent_type", "selected_tools", "user_request", "tool_hint", "error_feedback"],
    template="""
You are an expert Python developer. Generate a complete, runnable Python agent.

AGENT TYPE: {agent_type}
TOOLS: {selected_tools}
USER REQUEST: {user_request}
TOOL HINT: {tool_hint}

{error_feedback}

CRITICAL RULES:
- You MUST use ONLY Google Gemini via langchain_google_genai
- Model must be: gemini-2.5-flash

- DO NOT use OpenAI, HuggingFace, or any other models
- DO NOT use: langchain.chains, load_qa_chain
- DO NOT use: shutil, os.remove(), subprocess
- ONLY allowed libraries:
  - langchain_google_genai
  - langchain_chroma (if needed)
  - standard Python libraries

CODE REQUIREMENTS:
- Must define: class Agent
- Must implement: run(self, user_input: str) -> str
- Must include: if __name__ == "__main__"
- Must include proper try/except error handling
- Output must be fully runnable Python code

OUTPUT FORMAT:
Return ONLY Python code inside ```python``` block.
NO explanations, NO text outside code.
"""
)

class CodeGenerator:
    def __init__(self, max_retries: int = 2, backoff_base: float = 1.0):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            request_timeout=60
        )

        self.pattern_db = PatternDatabase()
        self.detector = ToolDetector()
        self.sandbox = AdvancedSandbox()
        self.scorer = CodeQualityScorer()
        self.cache = CodeCache()

        self.max_retries = max_retries
        self.backoff_base = backoff_base

        self.stats = {
            "llm_success": 0,
            "llm_failure": 0,
            "template_used": 0,
            "pattern_matches": 0,
            "retries": 0,
            "cache_hits": 0,
            "self_corrections": 0
        }

    def generate(self, plan: Dict, tool_result: Dict, user_request: str, constraints: Dict = None) -> Dict:
        """Generate code with self-correction loop"""
        agent_type = plan.get("agent_type", "chatbot")
        tools = self.detector.detect(user_request, plan.get("tools", []))
        plan["tools"] = tools
        constraints = constraints or {}

        # 1. Cache check
        cached = self.cache.get(user_request, agent_type, tools, constraints)
        if cached:
            self.stats["cache_hits"] += 1
            return self._save_cached_result(cached, agent_type, user_request)

        # 2. Pattern check
        pattern_match = self.pattern_db.find_matching_pattern(user_request, agent_type, tools)
        if pattern_match:
            pattern, _ = pattern_match
            if pattern.should_use_llm is False:
                logger.info("Pattern suggests template")
                result = self._fallback_to_template(
                    agent_type, tool_result, user_request,
                    method="pattern_template", pattern_used=True
                )
                self.cache.put(user_request, agent_type, tools, constraints, result)
                return result

        # 3. Self-correcting generation loop
        error_feedback = ""
        
        # Ensure generated_agents directory exists
        os.makedirs("generated_agents", exist_ok=True)
        
        for attempt in range(self.max_retries):
            logger.info(f"Self-correction attempt {attempt + 1}/{self.max_retries}")
            
            # Generate code with feedback
            code, err = self._generate_with_feedback(plan, tool_result, user_request, error_feedback)
            
            if not code:
                logger.warning(f"Generation failed: {err}")
                error_feedback = f"Generation Error: {err}\nPlease fix the code.\n"
                continue
            
            # Sandbox validation
            is_valid, msg = self.sandbox.validate(code)
            if not is_valid:
                logger.warning(f"Sandbox validation failed: {msg}")
                error_feedback = f"Validation Error: {msg}\nPlease fix the code.\n"
                continue
            
            # Save to temp file for testing
            temp_file = f"generated_agents/temp_agent_{attempt}.py"
            
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                # Import and test the module
                spec = importlib.util.spec_from_file_location(f"temp_agent_{attempt}", temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for Agent class
                if not hasattr(module, "Agent"):
                    raise Exception("Missing Agent class")
                
                # Test run method
                agent = module.Agent()
                test_output = agent.run("test input")
                
                if not isinstance(test_output, str):
                    raise Exception(f"run() returned {type(test_output)}, expected string")
                
                # All tests passed!
                quality = self.scorer.score(code)
                logger.info(f"Self-correction succeeded on attempt {attempt + 1}")
                self.stats["llm_success"] += 1
                self.stats["self_corrections"] = attempt
                
                result = self._save_and_record(code, agent_type, user_request, tool_result, 
                                            f"self_corrected_{attempt + 1}", quality, pattern_match is not None)
                self.cache.put(user_request, agent_type, tools, constraints, result)
                
                # Cleanup temp file
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Execution test failed: {error_msg}")
                error_feedback = f"Execution Error: {error_msg}\nPlease fix the code. Make sure:\n1. Agent class exists\n2. run() returns a string\n3. No missing imports\n"
                # Cleanup temp file on failure
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
                continue
        
        # 4. Fallback to template
        logger.warning(f"Self-correction failed after {self.max_retries} attempts, using template fallback")
        self.stats["llm_failure"] += 1
        
        result = self._fallback_to_template(agent_type, tool_result, user_request, 
                                            method="template_fallback", pattern_used=False)
        self.cache.put(user_request, agent_type, tools, constraints, result)
        return result

    def _generate_with_feedback(self, plan: Dict, tool_result: Dict, user_request: str, error_feedback: str = "") -> Tuple[Optional[str], Optional[str]]:
        """Generate code with optional error feedback"""
        
        selected = tool_result.get("selected_tools", {})
        tool_hint = "\n".join([f"- {k}: {v.get('implementation', '')}" for k, v in selected.items()])
        
        # Prepare feedback section
        feedback_section = ""
        if error_feedback:
            feedback_section = f"""
PREVIOUS ATTEMPT FAILED WITH:
{error_feedback}

Please fix the code and try again.
"""
        
        try:
            response = self.llm.invoke(
                SELF_CORRECTING_PROMPT.format(
                    agent_type=plan["agent_type"],
                    selected_tools=json.dumps(selected, indent=2),
                    user_request=user_request,
                    tool_hint=tool_hint or "None",
                    error_feedback=feedback_section
                )
            )
            
            raw = response.content
            
            # Handle response format
            if isinstance(raw, list):
                text_parts = []
                for part in raw:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                    else:
                        text_parts.append(str(part))
                raw = " ".join(text_parts)
            elif isinstance(raw, dict):
                if 'text' in raw:
                    raw = raw['text']
                else:
                    raw = str(raw)
            elif not isinstance(raw, str):
                raw = str(raw)
            
            # Clean markdown
            if "```python" in raw:
                raw = raw.split("```python")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            
            raw = raw.strip()
            
            if "class Agent" not in raw or "def run" not in raw:
                return None, "Missing Agent structure"
            
            return raw, None
            
        except Exception as e:
            return None, str(e)

    def _fallback_to_template(self, agent_type: str, tool_result: Dict,
                              user_request: str, method: str, pattern_used: bool) -> Dict:

        self.stats["template_used"] += 1

        if pattern_used:
            self.stats["pattern_matches"] += 1

        code = self._build_template(agent_type, tool_result, user_request, method)
        quality = self.scorer.score(code)

        return self._save_and_record(
            code, agent_type, user_request, tool_result,
            method, quality, pattern_used
        )

    def _build_template(self, agent_type: str, tool_result: Dict,
                         user_request: str, method: str) -> str:

        selected = tool_result.get("selected_tools", {})
        tool_code, tool_registry = self._build_tool_code(selected)

        timestamp = datetime.now().isoformat()

        if agent_type == "rag":
         return f'''"""
RAG AGENT with Document Upload (Fixed)
Generated: {timestamp}
Request: {user_request}
Method: {method}
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

class Agent:
    def __init__(self):
        # No api_key parameter needed - uses GOOGLE_API_KEY from environment
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        self.vectorstore = None
        self._load_existing_db()
    
    def _load_existing_db(self):
        if os.path.exists("./chroma_db"):
            try:
                self.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
                return True
            except Exception as e:
                print(f"Error loading DB: {e}")
                return False
        return False
    
    def add_documents(self, uploaded_files):
        documents = []
        
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding='utf-8')
                
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {uploaded_file.name}: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        if not documents:
            return 0
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
            
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        
        return len(chunks)
    
    def run(self, query: str) -> str:
        if not self.vectorstore:
            return "No documents loaded. Please upload PDF or TXT files first."
        
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            if not docs:
                return f"No relevant information found for: {{query}}"
            
            context = "\\n\\n".join([d.page_content for d in docs])
            prompt = f"Based on the following context, answer the question.\\n\\nContext:\\n{{context}}\\n\\nQuestion: {{query}}\\n\\nAnswer:"
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {{str(e)}}


def run_ui():
    st.set_page_config(page_title="RAG Agent", page_icon="📚", layout="wide")
    
    st.title("📚 Document Q&A Agent")
    st.markdown("Upload PDF or TXT documents and ask questions about their content")
    
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = Agent()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📤 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                chunk_count = st.session_state.rag_agent.add_documents(uploaded_files)
                if chunk_count > 0:
                    st.success(f"✅ Processed {{len(uploaded_files)}} files into {{chunk_count}} chunks")
                    st.rerun()
                else:
                    st.error("No documents could be processed")
        
        if st.session_state.rag_agent.vectorstore:
            st.info("✅ Documents are loaded and ready for questions")
    
    with col2:
        st.subheader("2. Ask Questions")
        query = st.text_area("Enter your question:", placeholder="What is this document about?")
        
        if st.button("🔍 Ask", type="primary") and query:
            with st.spinner("Searching for answer..."):
                response = st.session_state.rag_agent.run(query)
                st.markdown("### Answer")
                st.write(response)


if __name__ == "__main__":
    run_ui()
'''

        elif agent_type == "tool_agent":
            return f'''"""
TOOL AGENT (Template Fallback)
Generated: {timestamp}
Request: {user_request}
Method: {method}
"""
{tool_code}

class Agent:
    def __init__(self):
        self.tools = {{
            {tool_registry}
        }}

    def _detect_intent(self, inp: str) -> str:
        if any(op in inp for op in '+-*/'):
            return 'calculator'
        if any(w in inp.lower() for w in ['search', 'find', 'what is']):
            return 'search'
        return 'chat'

    def run(self, inp: str) -> str:
        intent = self._detect_intent(inp)
        if intent in self.tools:
            return self.tools[intent](inp)
        return f"Available tools: {{list(self.tools.keys())}}\\nYour request: {{inp}}"

if __name__ == "__main__":
    agent = Agent()
    print("Tool Agent Ready")
    print(f"Tools: {{list(agent.tools.keys())}}")
    while True:
        u = input("\\nYou: ")
        if u.lower() == "exit":
            break
        print(f"Agent: {{agent.run(u)}}")
'''

        else:
            return f'''"""
CHATBOT AGENT (Template Fallback)
Generated: {timestamp}
Request: {user_request}
Method: {method}
"""

class Agent:
    def __init__(self):
        self.history = []

    def run(self, inp: str) -> str:
        low = inp.lower()
        if any(g in low for g in ['hi', 'hello', 'hey']):
            resp = "Hello! How can I help you today?"
        elif any(b in low for b in ['bye', 'exit', 'quit']):
            resp = "Goodbye! Have a great day!"
        else:
            resp = f"You said: {{inp}}\\n\\nHow can I assist you?"

        self.history.append((inp, resp))
        return resp

if __name__ == "__main__":
    agent = Agent()
    print("Chatbot Ready. Type 'exit' to quit.")
    while True:
        u = input("\\nYou: ")
        if u.lower() == "exit":
            print("Goodbye!")
            break
        print(f"Bot: {{agent.run(u)}}")
'''

    def _build_tool_code(self, selected: Dict):
        code_parts, registry = [], []

        for tname, tdata in selected.items():
            impl = tdata.get("implementation")

            if tname in TOOL_IMPLEMENTATIONS and impl in TOOL_IMPLEMENTATIONS[tname]:
                code_parts.append(TOOL_IMPLEMENTATIONS[tname][impl])

                if tname == "search":
                    registry.append(f'"{tname}": search_tool')
                elif tname == "calculator":
                    registry.append(f'"{tname}": calculate_tool')
                elif tname == "retriever":
                    registry.append(f'"{tname}": retriever_tool')
                elif tname == "memory":
                    registry.append(f'"{tname}": MemoryTool()')

        return "\n\n".join(code_parts), ",\n            ".join(registry) if registry else '"chat": lambda x: x'

    def _save_and_record(self, code, agent_type, user_request,
                         tool_result, method, quality, pattern_used):

        os.makedirs("generated_agents", exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"generated_agents/{agent_type}_agent_{ts}.py"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

        tools = list(tool_result.get("selected_tools", {}).keys())

        self.pattern_db.record_result(
            user_request, agent_type, tools,
            method, True, quality
        )

        return {
            "filename": filename,
            "code": code,
            "lines": len(code.splitlines()),
            "agent_type": agent_type,
            "generation_method": method,
            "pattern_match_used": pattern_used,
            "quality_score": quality,
            "stats": self.stats.copy(),
            "timestamp": ts,
            "generation_source": method
        }

    def _save_cached_result(self, cached, agent_type, user_request):
        code = cached["code"]

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"generated_agents/{agent_type}_agent_{ts}.py"

        os.makedirs("generated_agents", exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

        return {
            "filename": filename,
            "code": code,
            "lines": len(code.splitlines()),
            "agent_type": agent_type,
            "generation_method": cached.get("generation_method", "cached"),
            "quality_score": cached.get("quality_score", 0.8),
            "cached": True,
            "timestamp": ts,
            "generation_source": "cache"
        }