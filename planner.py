from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import logging
import re
import ast

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an expert AI system planner.

Your task is to convert a user request into a structured agent plan.

-------------------------
AVAILABLE AGENT TYPES:
- chatbot: simple conversation (use memory for multi-turn)
- rag: document-based retrieval system
- tool_agent: uses external tools

-------------------------
AVAILABLE TOOLS:
- search: for web/information lookup
- calculator: for mathematical operations
- retriever: for document retrieval (RAG)
- memory: for conversation context, preferences, session state

Tool usage rules:
- Use search for external knowledge
- Use calculator only for math
- Use retriever for documents/PDFs
- Use memory when conversation continuity is needed
- Tools can be chained (e.g., search → calculator → search)

-------------------------
FLOW RULES:
- Flow must start with "parse_query" and end with "return_output"
- Use detailed step names
- Include tool selection steps explicitly
- For tool_agent:
  - "select_tool:X" must come before "execute_X"
- Every tool used in flow MUST be present in "tools"
- Do not reference tools in flow that are not listed
- If memory is used, include steps like "store_memory" or "retrieve_memory"

-------------------------
EDGE CASE HANDLING:
- If input is empty → chatbot
- If input is vague → chatbot
- If user requests documents AND web → tool_agent with ["retriever", "search"]
- If user requests calculation only → calculator only
- If unsure → chatbot

-------------------------
STRICT OUTPUT RULES:
- Return ONLY valid JSON
- JSON must contain EXACTLY: ["agent_type", "tools", "flow"]
- No extra keys
- No explanation, no markdown

-------------------------
EXAMPLES:

User: Build a chatbot for FAQs
Output:
{{"agent_type": "chatbot", "tools": [], "flow": ["parse_query", "generate_response", "return_output"]}}

User: Build a PDF QA system
Output:
{{"agent_type": "rag", "tools": ["retriever"], "flow": ["parse_query", "retrieve_documents", "generate_answer", "return_output"]}}

User: Build AI that searches and calculates
Output:
{{"agent_type": "tool_agent", "tools": ["search", "calculator"], "flow": ["parse_query", "select_tool:search", "execute_search", "select_tool:calculator", "compute", "synthesize", "return_output"]}}

-------------------------
USER INPUT:
\"\"\"
{user_input}
\"\"\"

OUTPUT:
"""
)

def extract_json(text) -> str:
    """
    Extract JSON from Gemini 3 response.
    Returns ONLY the JSON string, no extra fields.
    """
    # If it's a dict, extract the 'text' field
    if isinstance(text, dict):
        if 'text' in text:
            text = text['text']
        else:
            # If no 'text' field, try to convert the whole dict to JSON
            try:
                return json.dumps(text)
            except:
                return str(text)
    
    # If it's a list, join it
    if isinstance(text, list):
        text = " ".join(str(part) for part in text)
    
    # Ensure string
    if not isinstance(text, str):
        text = str(text)
    
    # Try to parse as Python literal (handles single quotes)
    if text.startswith("{'") or text.startswith('{"'):
        try:
            python_dict = ast.literal_eval(text)
            return json.dumps(python_dict)
        except:
            pass
    
    # Remove markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    # Find first { to last }
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start != -1 and end > start:
        json_str = text[start:end]
        # Fix single quotes to double quotes if needed
        if "'" in json_str and '"' not in json_str:
            json_str = json_str.replace("'", '"')
        return json_str
    
    return text.strip()

def is_vague_input(user_input: str) -> bool:
    if not user_input or len(user_input.strip()) == 0:
        return True
    words = user_input.split()
    if len(words) < 3:
        return True
    vague_patterns = [
        r"^(make|build|create)\s+(an?\s+)?(ai|agent|system)?\s*$",
        r"^(help|assist|do)\s+(me\s+)?(with\s+)?(something|anything)?\s*$",
        r"^(i\s+)?(need|want)\s+(an?\s+)?(ai|agent)\s*$"
    ]
    for pattern in vague_patterns:
        if re.search(pattern, user_input.lower()):
            return True
    return False

def validate_and_repair_plan(plan: dict, user_input: str = "") -> dict:
    valid_types = ["chatbot", "rag", "tool_agent"]
    valid_tools = {"search", "calculator", "retriever", "memory"}
    
    # Create clean output dict (only required keys)
    clean_plan = {}
    
    # Validate agent_type
    agent_type = plan.get("agent_type")
    if agent_type not in valid_types:
        logger.warning(f"Invalid agent_type '{agent_type}', defaulting to 'chatbot'")
        agent_type = "chatbot"
    clean_plan["agent_type"] = agent_type
    
    # Validate tools
    tools = plan.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    clean_tools = [t for t in tools if t in valid_tools]
    clean_tools = list(dict.fromkeys(clean_tools))
    clean_plan["tools"] = clean_tools
    
    # Validate flow
    flow = plan.get("flow", [])
    if not isinstance(flow, list) or len(flow) == 0:
        flow = ["parse_query", "generate_response", "return_output"]
    
    # Remove duplicate steps while preserving order
    seen = set()
    clean_flow = []
    for step in flow:
        if step not in seen:
            clean_flow.append(step)
            seen.add(step)
    
    # Ensure flow boundaries
    if clean_flow[0] != "parse_query":
        clean_flow.insert(0, "parse_query")
    
    if clean_flow[-1] != "return_output":
        clean_flow.append("return_output")
    
    # Fix tool-flow consistency
    fixed_flow = []
    for step in clean_flow:
        if step.startswith("select_tool:"):
            tool = step.split(":")[1]
            if tool in clean_tools:
                fixed_flow.append(step)
        elif step.startswith("execute_"):
            tool = step.replace("execute_", "")
            if tool in clean_tools:
                fixed_flow.append(step)
        else:
            fixed_flow.append(step)
    clean_plan["flow"] = fixed_flow
    
    # Add version
    clean_plan["version"] = "v1"
    
    # Calculate confidence
    confidence = 0.9
    if len(tools) != len(clean_tools):
        confidence -= 0.2
    if agent_type != plan.get("agent_type"):
        confidence -= 0.1
    
    clean_plan["confidence"] = max(0.5, min(0.95, confidence))
    
    return clean_plan

def generate_plan(user_input: str) -> dict:
    """Main function to generate agent plan from user input"""
    if is_vague_input(user_input):
        logger.warning(f"Input too vague: '{user_input}', defaulting to chatbot")
        return {
            "agent_type": "chatbot",
            "tools": [],
            "flow": ["parse_query", "generate_response", "return_output"],
            "confidence": 0.6,
            "version": "v1"
        }
    
    formatted_prompt = prompt.format(user_input=user_input)
    
    for attempt in range(2):
        try:
            response = llm.invoke(formatted_prompt)
            content = response.content
            
            # Extract JSON string
            json_str = extract_json(content)
            
            # Parse JSON
            plan = json.loads(json_str)
            
            # Validate and clean
            clean_plan = validate_and_repair_plan(plan, user_input)
            
            logger.info(f"Generated plan: agent_type={clean_plan['agent_type']}, tools={clean_plan['tools']}, confidence={clean_plan['confidence']}")
            return clean_plan
            
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt + 1} failed: JSON decode error")
            if attempt == 1:
                logger.error(f"JSON decode error after 2 attempts: {e}")
                return {
                    "agent_type": "chatbot",
                    "tools": [],
                    "flow": ["parse_query", "generate_response", "return_output"],
                    "confidence": 0.5,
                    "version": "v1"
                }
            continue
            
        except Exception as e:
            logger.error(f"Planner error on attempt {attempt + 1}: {e}")
            if attempt == 1:
                return {
                    "agent_type": "chatbot",
                    "tools": [],
                    "flow": ["parse_query", "generate_response", "return_output"],
                    "confidence": 0.5,
                    "version": "v1"
                }
            continue
    
    return {
        "agent_type": "chatbot",
        "tools": [],
        "flow": ["parse_query", "generate_response", "return_output"],
        "confidence": 0.5,
        "version": "v1"
    }

if __name__ == "__main__":
    test_inputs = [
        "Build a PDF QA system",
        "Build AI that searches then calculates",
        "hi",
    ]
    
    for inp in test_inputs:
        print(f"\n{'='*50}")
        print(f"Input: {inp}")
        result = generate_plan(inp)
        print(json.dumps(result, indent=2))