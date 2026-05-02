# test_pipeline.py

import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from planner import generate_plan
from TOOL_generator.tool_generator import ToolSelector

# Import from correct paths based on your structure
from code_generator.stage_3_code_generator import CodeGenerator
from Tester_agent.stage_4_tester import Stage4Tester
from Delpoy_agent.stage_5_deployer import Stage5Deployer


def run_full_pipeline(user_input: str):
    """Run complete pipeline from planning to deployment"""
    
    print("\n" + "="*80)
    print(f"USER INPUT: {user_input}")
    print("="*80)

    # -------------------------
    # Stage 1: Planner
    # -------------------------
    print("\n[STAGE 1] PLANNER")
    print("-"*50)
    plan = generate_plan(user_input)
    print(json.dumps(plan, indent=2))
    
    assert "agent_type" in plan
    assert "tools" in plan

    # -------------------------
    # Stage 2: Tool Selector
    # -------------------------
    print("\n[STAGE 2] TOOL SELECTOR")
    print("-"*50)
    selector = ToolSelector()
    tool_result = selector.select_tools(plan, user_input)
    print(json.dumps(tool_result, indent=2))
    
    assert "selected_tools" in tool_result
    assert tool_result.get("ready_for_stage_3", False) is True

    # -------------------------
    # Stage 3: Code Generator
    # -------------------------
    print("\n[STAGE 3] CODE GENERATOR")
    print("-"*50)
    generator = CodeGenerator()
    code_result = generator.generate(plan, tool_result, user_input)
    
    print(f"File Generated: {code_result['filename']}")
    print(f"Method: {code_result['generation_method']}")
    print(f"Quality Score: {code_result['quality_score']}")
    print(f"Lines: {code_result['lines']}")

    if "stats" in code_result:
        print(f"Generator Stats: {json.dumps(code_result['stats'], indent=2)}")

    # -------------------------
    # Stage 4: Tester
    # -------------------------
    print("\n[STAGE 4] TESTER")
    print("-"*50)
    tester = Stage4Tester()
    test_result = tester.run_tests(code_result["filename"])
    
    print(f"Tests Passed: {test_result['passed']}")
    for res in test_result.get("test_results", []):
        print(f"  {res}")
    for warn in test_result.get("warnings", []):
        print(f"  {warn}")

    if not test_result["passed"]:
        print(f"Test Error: {test_result.get('error', 'Unknown error')}")
        print("Pipeline stopped - cannot deploy failed agent")
        return None

    # -------------------------
    # Stage 5: Deployer
    # -------------------------
    print("\n[STAGE 5] DEPLOYER")
    print("-"*50)
    deployer = Stage5Deployer()
    deployment = deployer.deploy(
        tested_file=code_result["filename"],
        agent_name=plan.get("agent_type", "agent"),
        create_api=False
    )
    
    print(f"Deployment Success: {deployment['success']}")
    print(f"Deployed File: {deployment.get('deployed_file', 'N/A')}")
    print(f"Run Command: {deployment.get('run_command', 'N/A')}")

    # Quick test of deployed agent
    quick_test = deployer.quick_test(deployment)
    if quick_test.get("success"):
        print(f"Quick Test Output: {quick_test.get('test_output', '')[:200]}")
    else:
        print(f"Quick Test Failed: {quick_test.get('error', 'Unknown')}")

    print("\n" + "="*80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)

    return {
        "plan": plan,
        "tool_result": tool_result,
        "code_result": code_result,
        "test_result": test_result,
        "deployment": deployment
    }


def test_rag():
    """Test RAG agent - CHANGE THIS INPUT TO BYPASS CACHE"""
    # Change this string to something new to avoid cache hit
    user_input = "Build a PDF QA system that can answer questions from uploaded documents using vector search"  # ← Modified slightly
    return run_full_pipeline(user_input)


def test_chatbot():
    """Test chatbot agent"""
    user_input = "Build a friendly chatbot for customer service"
    return run_full_pipeline(user_input)


def test_calculator():
    """Test calculator agent"""
    user_input = "Build a calculator that can do basic arithmetic operations"
    return run_full_pipeline(user_input)


def test_search():
    """Test search agent"""
    user_input = "Build a web search assistant that can find information online"
    return run_full_pipeline(user_input)


# NEW: Test with completely different input to force fresh generation
def test_different():
    """Test with a different request - no cache match"""
    user_input = "Create a document question answering system using ChromaDB"
    return run_full_pipeline(user_input)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING FULL PIPELINE - RAG AGENT (MODIFIED INPUT)")
    print("="*80)
    
    try:
        # Option 1: Use modified RAG input
        result = test_rag()
        
        # Option 2: Or use completely different input
        # result = test_different()
        
        if result:
            print("\n[SUMMARY]")
            print(f"  Agent Type: {result['plan']['agent_type']}")
            print(f"  Generation Method: {result['code_result']['generation_method']}")
            print(f"  Quality Score: {result['code_result']['quality_score']}")
            print(f"  Tests Passed: {result['test_result']['passed']}")
            print(f"  Deployed: {result['deployment']['success']}")
            print(f"  Run: {result['deployment']['run_command']}")
        else:
            print("\n[FAILED] Pipeline did not complete successfully")
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()