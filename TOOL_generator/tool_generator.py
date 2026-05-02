# stage_2_tool_selector.py - WITH WHAT-IF SIMULATION

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Tool Registry - Database of available implementations
TOOL_REGISTRY = {
    "search": {
        "implementations": [
            {
                "id": "duckduckgo",
                "name": "DuckDuckGo Search",
                "description": "Free, no API key required",
                "pros": ["Free", "No API key", "Privacy focused"],
                "cons": ["Rate limited"],
                "cost": "Free",
                "api_key_required": False,
                "best_for": ["Prototyping", "Hackathons"],
                "setup_time": "2 minutes",
                "performance": "Medium"
            },
            {
                "id": "tavily",
                "name": "Tavily AI Search",
                "description": "LLM-optimized search API",
                "pros": ["High quality", "LLM-optimized"],
                "cons": ["Paid after free tier"],
                "cost": "Paid (free tier available)",
                "api_key_required": True,
                "best_for": ["Production", "Research"],
                "setup_time": "5 minutes",
                "performance": "High"
            }
        ]
    },
    "calculator": {
        "implementations": [
            {
                "id": "safe_eval",
                "name": "Python Safe Eval",
                "description": "AST-based safe evaluation",
                "pros": ["Safe", "No dependencies"],
                "cons": ["Basic math only"],
                "cost": "Free",
                "api_key_required": False,
                "best_for": ["Basic math", "Educational"],
                "setup_time": "0 minutes",
                "performance": "High"
            },
            {
                "id": "numexpr",
                "name": "NumExpr",
                "description": "Fast numerical evaluation",
                "pros": ["Very fast", "Efficient"],
                "cons": ["External dependency"],
                "cost": "Free",
                "api_key_required": False,
                "best_for": ["Performance-critical"],
                "setup_time": "2 minutes",
                "performance": "Very High"
            }
        ]
    },
    "retriever": {
        "implementations": [
            {
                "id": "chromadb",
                "name": "ChromaDB",
                "description": "Lightweight vector database",
                "pros": ["Easy setup", "Persistent"],
                "cons": ["Memory intensive"],
                "cost": "Free",
                "api_key_required": False,
                "best_for": ["Prototyping", "Small-medium datasets"],
                "setup_time": "3 minutes",
                "performance": "Medium"
            },
            {
                "id": "faiss",
                "name": "FAISS",
                "description": "Facebook AI Similarity Search",
                "pros": ["Very fast", "Memory efficient"],
                "cons": ["No built-in persistence"],
                "cost": "Free",
                "api_key_required": False,
                "best_for": ["High performance", "Large datasets"],
                "setup_time": "5 minutes",
                "performance": "Very High"
            },
            {
                "id": "pinecone",
                "name": "Pinecone",
                "description": "Managed vector database",
                "pros": ["Scalable", "Cloud-native"],
                "cons": ["Paid", "API key required"],
                "cost": "Paid",
                "api_key_required": True,
                "best_for": ["Production", "Large scale"],
                "setup_time": "10 minutes",
                "performance": "High"
            }
        ]
    },
    "memory": {
        "implementations": [
            {
                "id": "buffer_memory",
                "name": "Buffer Memory",
                "description": "Simple in-memory storage",
                "pros": ["Simple", "No setup"],
                "cons": ["No persistence"],
                "cost": "Free",
                "api_key_required": False,
                "best_for": ["Prototyping", "Simple chatbots"],
                "setup_time": "0 minutes",
                "performance": "High"
            },
            {
                "id": "redis_memory",
                "name": "Redis Memory",
                "description": "Persistent Redis-backed memory",
                "pros": ["Persistent", "Scalable"],
                "cons": ["Requires Redis server"],
                "cost": "Free (self-hosted)",
                "api_key_required": False,
                "best_for": ["Production", "Multi-user"],
                "setup_time": "15 minutes",
                "performance": "High"
            }
        ]
    }
}


class ToolSelector:
    """Stage 2: Tool Selector - Selects tools with what-if simulation"""
    
    def __init__(self):
        self.registry = TOOL_REGISTRY
        self.version = "2.1.0"
        self.fallback_count = 0
    
    def simulate_options(self, tool_name: str, constraints: Dict) -> Dict:
        """
        What-If Simulation: Compare all tool options before selection
        
        Args:
            tool_name: The tool to simulate (search, calculator, retriever, memory)
            constraints: User constraints (budget, privacy, performance, etc.)
        
        Returns:
            Dict with all options, scores, and recommendation
        """
        if tool_name not in self.registry:
            return {
                "tool": tool_name,
                "error": f"Tool '{tool_name}' not found in registry",
                "simulated_options": [],
                "recommended": None
            }
        
        options = []
        implementations = self.registry[tool_name]["implementations"]
        
        for impl in implementations:
            score = 0
            score_breakdown = {}
            
            # Budget constraint
            if constraints.get("budget") == "free":
                if impl["cost"] == "Free":
                    score += 30
                    score_breakdown["budget"] = 30
                elif "free tier" in impl["cost"].lower():
                    score += 20
                    score_breakdown["budget"] = 20
                else:
                    score_breakdown["budget"] = 0
            elif constraints.get("budget") == "paid":
                if impl["cost"] != "Free":
                    score += 20
                    score_breakdown["budget"] = 20
                else:
                    score_breakdown["budget"] = 10
            
            # Privacy constraint
            if constraints.get("privacy") == "strict":
                if not impl.get("api_key_required", False):
                    score += 30
                    score_breakdown["privacy"] = 30
                else:
                    score_breakdown["privacy"] = 0
            else:
                score_breakdown["privacy"] = 15
            
            # Performance preference
            if constraints.get("performance") == "fast":
                if impl.get("performance") == "Very High":
                    score += 25
                    score_breakdown["performance"] = 25
                elif impl.get("performance") == "High":
                    score += 15
                    score_breakdown["performance"] = 15
                else:
                    score_breakdown["performance"] = 5
            else:
                score_breakdown["performance"] = 10
            
            # Setup time preference
            if constraints.get("setup_time") == "quick":
                setup_min = 0
                if "minutes" in impl.get("setup_time", "5 minutes"):
                    try:
                        setup_min = int(impl.get("setup_time", "5").split()[0])
                    except:
                        setup_min = 5
                if setup_min <= 2:
                    score += 15
                    score_breakdown["setup_time"] = 15
                elif setup_min <= 5:
                    score += 10
                    score_breakdown["setup_time"] = 10
                else:
                    score_breakdown["setup_time"] = 5
            
            options.append({
                "id": impl["id"],
                "name": impl["name"],
                "cost": impl["cost"],
                "api_key_required": impl["api_key_required"],
                "setup_time": impl.get("setup_time", "Unknown"),
                "performance": impl.get("performance", "Medium"),
                "pros": impl["pros"],
                "cons": impl["cons"],
                "score": score,
                "score_breakdown": score_breakdown
            })
        
        # Sort by score (highest first)
        options.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "tool": tool_name,
            "simulated_options": options,
            "recommended": options[0]["id"] if options else None,
            "recommended_name": options[0]["name"] if options else None,
            "reasoning": f"Selected {options[0]['name']} with score {options[0]['score']}/100 based on constraints: {json.dumps(constraints)}"
        }
    
    def select_tools(self, planner_output: Dict, user_request: str = None, constraints: Dict = None) -> Dict:
        """
        Select best tool implementations based on plan and constraints
        Uses what-if simulation to compare options
        
        Args:
            planner_output: Output from Stage 1 (contains 'tools' list)
            user_request: Original user request for context
            constraints: Optional constraints like {'budget': 'free', 'use_case': 'prototype'}
        
        Returns:
            Dict with selected tool implementations and what-if simulation results
        """
        
        required_tools = planner_output.get('tools', [])
        agent_type = planner_output.get('agent_type', 'chatbot')
        
        logger.info(f"Stage 2: Tool Selector")
        logger.info(f"   Agent Type: {agent_type}")
        logger.info(f"   Tools needed: {required_tools}")
        
        fallback_used = False
        
        if not required_tools:
            return {
                "stage": "stage_2_tool_selection",
                "stage_number": 2,
                "next_stage": "stage_3_code_generation",
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "selected_tools": {},
                "simulations": {},
                "message": "No tools required for this agent",
                "ready_for_stage_3": True,
                "execution_mode": {
                    "mode": "hybrid",
                    "code_execution": "sub_agent_only",
                    "meta_agent_execution": "prebuilt_only",
                    "description": "Meta-agent selects tools, sub-agent executes them"
                }
            }
        
        # Default constraints
        default_constraints = {
            "budget": "free",
            "privacy": "moderate",
            "performance": "balanced",
            "setup_time": "quick",
            "use_case": "prototype" if agent_type != "tool_agent" else "general",
            "production_ready": False
        }
        
        if constraints:
            default_constraints.update(constraints)
        
        # Run what-if simulations for each required tool
        simulations = {}
        for tool in required_tools:
            simulations[tool] = self.simulate_options(tool, default_constraints)
        
        # Prepare rich context for LLM
        context = {
            "agent_type": agent_type,
            "request": user_request,
            "constraints": default_constraints,
            "required_tools": required_tools,
            "planner_output": planner_output,
            "simulations": simulations
        }
        
        # Use LLM to select tools (with simulation results as input)
        try:
            selections = self._llm_select_tools(required_tools, agent_type, default_constraints, user_request, context, simulations)
        except Exception as e:
            logger.error(f"LLM selection failed: {e}, using fallback")
            selections = self._fallback_selection(required_tools)
            fallback_used = True
        
        # Add implementation details from registry and simulation results
        enriched_tools = {}
        for tool_name, selection in selections.get("tools", {}).items():
            impl_id = selection.get("implementation")
            impl_details = self._get_implementation_details(tool_name, impl_id)
            
            # Get simulation result for this tool
            sim_result = simulations.get(tool_name, {})
            recommended = sim_result.get("recommended")
            
            if impl_details:
                enriched_tools[tool_name] = {
                    "implementation": impl_id,
                    "reasoning": selection.get("reasoning", "Selected based on constraints"),
                    "was_recommended": impl_id == recommended,
                    "details": {
                        "id": impl_details["id"],
                        "name": impl_details["name"],
                        "cost": impl_details["cost"],
                        "api_key_required": impl_details["api_key_required"],
                        "best_for": impl_details["best_for"],
                        "install_command": impl_details.get("install", "pip install [package]"),
                        "setup_time": impl_details.get("setup_time", "Unknown"),
                        "performance": impl_details.get("performance", "Medium")
                    }
                }
            else:
                enriched_tools[tool_name] = selection
        
        # Build result with enhanced metadata
        result = {
            "stage": "stage_2_tool_selection",
            "stage_number": 2,
            "next_stage": "stage_3_code_generation",
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "selected_tools": enriched_tools,
            "what_if_simulations": simulations,
            "summary": selections.get("summary", {
                "total_cost": "Free",
                "api_keys_needed": [],
                "setup_complexity": "Low"
            }),
            "original_plan": planner_output,
            "ready_for_stage_3": True,
            "execution_mode": {
                "mode": "hybrid",
                "code_execution": "sub_agent_only",
                "meta_agent_execution": "prebuilt_only",
                "description": "Meta-agent selects tools, sub-agent executes them",
                "why": "Ensures meta-agent reliability while allowing sub-agent flexibility"
            },
            "context_used": context,
            "fallback_used": fallback_used,
            "constraints_applied": default_constraints
        }
        
        if fallback_used:
            result["warning"] = "Fallback selection used due to LLM error. Review selections if needed."
            logger.warning(f"Fallback used for tool selection")
        
        logger.info(f"Selected implementations: {list(enriched_tools.keys())}")
        logger.info(f"Total cost: {result['summary']['total_cost']}")
        logger.info(f"API keys needed: {result['summary']['api_keys_needed']}")
        
        return result
    
    def _llm_select_tools(self, required_tools: List[str], agent_type: str, constraints: Dict, 
                          user_request: str, context: Dict, simulations: Dict) -> Dict:
        """Use LLM to intelligently select tools with simulation results"""
        
        # Build tools context for LLM
        tools_context = {}
        for tool in required_tools:
            if tool in self.registry:
                impls = self.registry[tool]["implementations"]
                tools_context[tool] = [
                    {
                        "id": impl["id"],
                        "name": impl["name"],
                        "cost": impl["cost"],
                        "api_key_required": impl["api_key_required"],
                        "pros": impl["pros"],
                        "cons": impl["cons"],
                        "best_for": impl["best_for"],
                        "performance": impl.get("performance", "Medium"),
                        "setup_time": impl.get("setup_time", "Unknown")
                    }
                    for impl in impls
                ]
        
        # Format simulations for LLM
        sim_summary = {}
        for tool, sim in simulations.items():
            sim_summary[tool] = {
                "recommended": sim.get("recommended"),
                "top_options": [
                    {"id": opt["id"], "score": opt["score"]}
                    for opt in sim.get("simulated_options", [])[:3]
                ]
            }
        
        selector_prompt = PromptTemplate(
            input_variables=["required_tools", "agent_type", "tools_context", "constraints", 
                           "user_request", "context", "simulations"],
            template="""
You are a tool selector for AI agents.

AGENT TYPE: {agent_type}
REQUIRED TOOLS: {required_tools}
USER REQUEST: {user_request}
CONSTRAINTS: {constraints}

WHAT-IF SIMULATION RESULTS (pre-computed scores):
{simulations}

RICH CONTEXT:
{context}

AVAILABLE TOOL IMPLEMENTATIONS:
{tools_context}

SELECTION RULES:
1. Use the simulation scores as primary guidance
2. Prioritize FREE tools unless production specified
3. Prioritize NO API KEY for prototypes
4. Match BEST_FOR with agent type and use case
5. Consider pros/cons for reliability

OUTPUT JSON ONLY (no other text):
{{
    "tools": {{
        "tool_name": {{
            "implementation": "selected_id",
            "reasoning": "why this choice (include simulation score if available)"
        }}
    }},
    "summary": {{
        "total_cost": "Free/Paid/Mixed",
        "api_keys_needed": ["key1", "key2"],
        "setup_complexity": "Low/Medium/High"
    }}
}}
"""
        )
        
        response = llm.invoke(selector_prompt.format(
            agent_type=agent_type,
            required_tools=json.dumps(required_tools),
            tools_context=json.dumps(tools_context, indent=2),
            constraints=json.dumps(constraints, indent=2),
            user_request=user_request or "Not specified",
            context=json.dumps(context, indent=2),
            simulations=json.dumps(sim_summary, indent=2)
        ))
        
        # Extract JSON
        import re
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError("Could not parse LLM response")
    
    def _fallback_selection(self, required_tools: List[str]) -> Dict:
        """Fallback selection (no LLM) with logging"""
        
        self.fallback_count += 1
        
        defaults = {
            "search": "duckduckgo",
            "calculator": "safe_eval",
            "retriever": "chromadb",
            "memory": "buffer_memory"
        }
        
        tools = {}
        for tool in required_tools:
            if tool in defaults:
                tools[tool] = {
                    "implementation": defaults[tool],
                    "reasoning": f"Default selection (fallback #{self.fallback_count}) - LLM unavailable"
                }
            else:
                tools[tool] = {
                    "implementation": "unknown",
                    "reasoning": f"Tool '{tool}' has no default implementation"
                }
        
        return {
            "tools": tools,
            "summary": {
                "total_cost": "Free",
                "api_keys_needed": [],
                "setup_complexity": "Low",
                "note": "Fallback selection used"
            }
        }
    
    def _get_implementation_details(self, tool_name: str, impl_id: str) -> Dict:
        """Get implementation details from registry"""
        if tool_name in self.registry:
            for impl in self.registry[tool_name]["implementations"]:
                if impl["id"] == impl_id:
                    return impl
        return None


if __name__ == "__main__":
    # Import Stage 1 (assuming it exists)
    try:
        from stage_1_planner import generate_plan
    except ImportError:
        # Mock planner for demo
        def generate_plan(request):
            if "PDF" in request or "document" in request:
                return {"agent_type": "rag", "tools": ["retriever"]}
            elif "search" in request:
                return {"agent_type": "tool_agent", "tools": ["search"]}
            elif "calculator" in request:
                return {"agent_type": "tool_agent", "tools": ["calculator"]}
            else:
                return {"agent_type": "chatbot", "tools": []}
    
    print("\n" + "="*70)
    print("STAGE 2: TOOL SELECTOR WITH WHAT-IF SIMULATION")
    print("Explicit stage tagging + execution mode + fallback tracking")
    print("="*70)
    
    test_requests = [
        "Build a PDF QA system",
        "Build a web search assistant",
        "Build a calculator with memory"
    ]
    
    selector = ToolSelector()
    
    for request in test_requests:
        print("\n" + "="*60)
        print(f"USER: {request}")
        print("="*60)
        
        plan = generate_plan(request)
        print("\nSTAGE 1 OUTPUT:")
        print(f"   Agent Type: {plan.get('agent_type')}")
        print(f"   Tools: {plan.get('tools')}")
        
        print("\nSTAGE 2 OUTPUT:")
        result = selector.select_tools(plan, request, {"budget": "free", "use_case": "prototype"})
        
        print(f"\n   Stage: {result['stage']}")
        print(f"   Next: {result['next_stage']}")
        
        print(f"\n   Execution Mode:")
        print(f"      Mode: {result['execution_mode']['mode']}")
        print(f"      Code Executes In: {result['execution_mode']['code_execution']}")
        print(f"      Meta-Agent Uses: {result['execution_mode']['meta_agent_execution']}")
        
        print(f"\n   What-If Simulation Results:")
        for tool_name, sim in result.get("what_if_simulations", {}).items():
            print(f"      {tool_name}:")
            print(f"         Recommended: {sim.get('recommended_name', sim.get('recommended'))}")
            for opt in sim.get("simulated_options", [])[:3]:
                print(f"         - {opt['name']}: score {opt['score']}/100 (cost: {opt['cost']})")
        
        print(f"\n   Selected Tools:")
        for tool_name, tool_data in result["selected_tools"].items():
            print(f"      {tool_name} -> {tool_data.get('implementation')}")
            print(f"        Reasoning: {tool_data.get('reasoning', 'N/A')}")
            if "details" in tool_data:
                print(f"        Cost: {tool_data['details'].get('cost')}")
                print(f"        API Key: {'Yes' if tool_data['details'].get('api_key_required') else 'No'}")
                print(f"        Recommended by simulation: {tool_data.get('was_recommended', False)}")
        
        print(f"\n   Summary:")
        print(f"      Total Cost: {result['summary'].get('total_cost')}")
        print(f"      API Keys Needed: {result['summary'].get('api_keys_needed')}")
        print(f"      Setup Complexity: {result['summary'].get('setup_complexity')}")
        
        print(f"\n   Fallback Used: {result.get('fallback_used', False)}")
        print(f"   Ready for Stage 3: {result['ready_for_stage_3']}")
        
        print(f"\n   Constraints Applied: {json.dumps(result.get('constraints_applied', {}), indent=2)}")
        
        input("\nPress Enter to continue...")
    
    print("\n" + "="*60)
    print("TOOL SELECTOR STATISTICS")
    print("="*60)
    print(f"   Total fallbacks used: {selector.fallback_count}")
    print(f"   Version: {selector.version}")
    print(f"   Tools in registry: {len(TOOL_REGISTRY)}")
    print(f"   Total implementations: {sum(len(v['implementations']) for v in TOOL_REGISTRY.values())}")
    
    print("\nStage 2 complete. Ready for Stage 3: Code Generation")