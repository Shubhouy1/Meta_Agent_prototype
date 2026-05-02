# code_generator/stage_4_tester.py

import os
import ast
import importlib.util
import traceback
import sys
import time
import threading
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Stage4Tester:
    """
    Stage 4: Tests generated agent code with fallback support
    Checks: syntax, imports, structure, sample execution
    Compatible with Windows and Unix
    """
    
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []
        self.test_timeout = 5  # seconds for sample execution
        self.original_recursion_limit = 1000
    
    # -------------------------
    # MAIN ENTRY
    # -------------------------
    def run_tests(self, file_path: str, use_fallback: bool = True) -> Dict:
        """
        Run all tests on generated agent file with fallback option
        
        Args:
            file_path: Path to generated agent Python file
            use_fallback: If True, can suggest fallback on import errors
            
        Returns:
            Dict with test results
        """
        self.results = []
        self.warnings = []
        self.errors = []
        
        print(f"\n🧪 Stage 4: Testing Agent")
        print(f"   File: {file_path}")
        
        if not os.path.exists(file_path):
            return {
                "passed": False,
                "error": f"File not found: {file_path}",
                "test_results": [],
                "warnings": [],
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Set recursion limit for safety
            sys.setrecursionlimit(1000)
            
            # 1. Syntax Check (with __main__ detection)
            self._syntax_check(file_path)
            
            # 2. Import Test (safe)
            module = self._import_test(file_path)
            
            # 3. Structure Check
            self._structure_check(module)
            
            # 4. Sample Execution (safe with timing)
            self._sample_execution(module)
            
            # Restore recursion limit
            sys.setrecursionlimit(self.original_recursion_limit)
            
            # All tests passed
            return {
                "passed": True,
                "test_results": self.results,
                "warnings": self.warnings,
                "errors": self.errors,
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "fallback_available": False
            }
            
        except ImportError as e:
            # Handle missing dependencies
            error_msg = str(e)
            missing_module = None
            
            # Extract missing module name
            if "No module named" in error_msg:
                missing_module = error_msg.split("'")[1]
                self.warnings.append(f"⚠️ Missing dependency: {missing_module}")
                self.warnings.append(f"💡 Run: pip install {missing_module.split('.')[0]}")
            
            # Restore recursion limit
            sys.setrecursionlimit(self.original_recursion_limit)
            
            return {
                "passed": False,
                "error": error_msg,
                "error_type": "import_error",
                "missing_module": missing_module,
                "traceback": traceback.format_exc(),
                "test_results": self.results,
                "warnings": self.warnings,
                "errors": self.errors,
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "fallback_available": use_fallback,
                "fallback_recommendation": "Use template fallback or install missing dependencies"
            }
            
        except Exception as e:
            # Restore recursion limit
            sys.setrecursionlimit(self.original_recursion_limit)
            
            return {
                "passed": False,
                "error": str(e),
                "error_type": "general_error",
                "traceback": traceback.format_exc(),
                "test_results": self.results,
                "warnings": self.warnings,
                "errors": self.errors,
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "fallback_available": use_fallback,
                "fallback_recommendation": "Use template fallback for reliable deployment"
            }
    
    # -------------------------
    # 1. SYNTAX CHECK (with file size warning + __main__ detection)
    # -------------------------
    def _syntax_check(self, file_path: str):
        """Check if code compiles without syntax errors"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            # Check for __main__ block (warning but not error)
            if '__main__' in code and 'if __name__ == "__main__"' in code:
                self.warnings.append("⚠️ __main__ block detected (safe but monitored)")
            
            # Check for potentially dangerous patterns (warning only)
            dangerous_patterns = [
                ('while True', "Potential infinite loop detected"),
                ('for ', "Loop detected (ensure it terminates)"),
                ('__import__', "Dynamic import detected"),
            ]
            
            for pattern, warning in dangerous_patterns:
                if pattern in code:
                    self.warnings.append(f"⚠️ {warning}")
                    break  # Report once
            
            # Parse AST
            ast.parse(code)
            self.results.append("✅ Syntax check passed")
            
            # Check line count
            line_count = len(code.splitlines())
            if line_count < 10:
                self.warnings.append(f"⚠️ Agent code is very short ({line_count} lines)")
            elif line_count > 500:
                self.warnings.append(f"⚠️ Code is very large ({line_count} lines) - may impact performance")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024:  # 100 KB
                self.warnings.append(f"⚠️ File size is large ({file_size // 1024} KB)")
                
        except SyntaxError as e:
            raise Exception(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            raise Exception(f"Syntax check failed: {str(e)}")
    
    # -------------------------
    # 2. IMPORT TEST (with safety guard)
    # -------------------------
    def _import_test(self, file_path: str):
        """Try to import the module safely"""
        module_name = os.path.basename(file_path).replace(".py", "")
        
        # Pre-read file for safety warnings
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_preview = f.read()
                
            # Check for suspicious patterns before import
            suspicious = [
                ('os.system', "Potential system command execution"),
                ('subprocess', "Subprocess module import may be dangerous"),
                ('eval(', "eval() usage can be risky"),
                ('exec(', "exec() usage can be risky"),
                ('__import__', "Dynamic import"),
                ('open(', "File access (may be legitimate)"),
            ]
            
            for pattern, warning in suspicious:
                if pattern in code_preview:
                    self.warnings.append(f"⚠️ {warning} pattern detected")
                    
        except Exception as e:
            self.warnings.append(f"⚠️ Could not pre-scan file: {e}")
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise Exception(f"Cannot create spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute with timeout protection using threading
            self._exec_with_timeout(spec.loader, module)
            
            self.results.append("✅ Import test passed")
            return module
            
        except ImportError as e:
            raise Exception(f"Import error: {e}")
        except Exception as e:
            raise Exception(f"Import test failed: {str(e)}")
    
    # -------------------------
    # 3. STRUCTURE CHECK
    # -------------------------
    def _structure_check(self, module):
        """Verify Agent class and run method exist"""
        
        # Check for Agent class
        if not hasattr(module, "Agent"):
            raise Exception("Missing Agent class in generated code")
        
        agent_class = getattr(module, "Agent")
        
        # Check it's a class
        if not isinstance(agent_class, type):
            raise Exception("Agent is not a class")
        
        # Check for run method
        if not hasattr(agent_class, "run"):
            raise Exception("Missing run() method in Agent class")
        
        run_method = getattr(agent_class, "run")
        
        # Check run is callable
        if not callable(run_method):
            raise Exception("run() is not callable")
        
        # Check method signature (optional, more advanced)
        import inspect
        sig = inspect.signature(run_method)
        params = list(sig.parameters.values())
        
        if len(params) < 1:
            self.warnings.append("⚠️ run() method should accept at least 'self' and 'user_input'")
        elif len(params) == 1:
            self.warnings.append("⚠️ run() method accepts only 'self' - should also accept user input")
        
        self.results.append("✅ Structure check passed")
    
    # -------------------------
    # 4. SAMPLE EXECUTION (with timing)
    # -------------------------
    def _sample_execution(self, module):
        """Run a sample test to verify basic functionality with timing"""
        
        try:
            # Create agent instance
            agent_class = getattr(module, "Agent")
            agent = agent_class()
            
            # Test with simple input
            test_input = "Hello, this is a test"
            
            # Execute with timeout (cross-platform) and timing
            result = {"output": None, "error": None, "execution_time": None}
            
            def run_in_thread():
                try:
                    start = time.time()
                    result["output"] = agent.run(test_input)
                    result["execution_time"] = time.time() - start
                except Exception as e:
                    result["error"] = str(e)
            
            # Start thread
            thread = threading.Thread(target=run_in_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.test_timeout)
            
            if thread.is_alive():
                raise Exception(f"Sample execution timed out after {self.test_timeout} seconds")
            
            if result["error"]:
                raise Exception(f"Execution error: {result['error']}")
            
            output = result["output"]
            exec_time = result["execution_time"]
            
            # Add timing to results
            if exec_time is not None:
                self.results.append(f"⏱ Execution time: {round(exec_time, 3)}s")
            
            # Check output type
            if not isinstance(output, str):
                self.warnings.append(f"⚠️ run() should return string, got {type(output).__name__}")
            
            # Check output is not empty
            if not output or len(output.strip()) == 0:
                self.warnings.append("⚠️ run() returned empty response")
            
            # Check execution time warning
            if exec_time and exec_time > 2.0:
                self.warnings.append(f"⚠️ Execution took {round(exec_time, 2)}s - may be slow")
            
            self.results.append(f"✅ Sample execution passed (response length: {len(output) if output else 0} chars)")
            
        except Exception as e:
            raise Exception(f"Sample execution failed: {str(e)}")
    
    # -------------------------
    # HELPER: Timeout for import
    # -------------------------
    def _exec_with_timeout(self, loader, module, timeout: int = 10):
        """Execute module loader with timeout (cross-platform)"""
        
        result = {"error": None, "completed": False}
        
        def exec_in_thread():
            try:
                loader.exec_module(module)
                result["completed"] = True
            except Exception as e:
                result["error"] = str(e)
        
        thread = threading.Thread(target=exec_in_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            raise Exception(f"Import timed out after {timeout} seconds")
        
        if result["error"]:
            raise Exception(result["error"])
        
        if not result["completed"]:
            raise Exception("Import failed to complete")
    
    # -------------------------
    # 5. EXTRA: TOOL AVAILABILITY CHECK (Optional)
    # -------------------------
    def check_tool_availability(self, file_path: str, expected_tools: List[str]) -> Dict:
        """
        Check if expected tools are available in the generated agent
        
        Args:
            file_path: Path to agent file
            expected_tools: List of tool names expected (e.g., ['search', 'calculator'])
            
        Returns:
            Dict with availability results
        """
        results = {}
        
        for tool in expected_tools:
            results[tool] = {
                "available": False,
                "check": f"Checking for {tool}..."
            }
        
        try:
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for tool functions in module or Agent class
            for tool in expected_tools:
                found = False
                
                # Check module level
                if hasattr(module, f"{tool}_tool"):
                    found = True
                    results[tool] = {"available": True, "location": "module"}
                
                # Check Agent class
                if hasattr(module, "Agent"):
                    agent_class = getattr(module, "Agent")
                    if hasattr(agent_class, f"{tool}_tool"):
                        found = True
                        results[tool] = {"available": True, "location": "Agent class"}
                
                if not found:
                    results[tool] = {"available": False, "warning": f"{tool} tool not found"}
                    self.warnings.append(f"⚠️ Expected tool '{tool}' not found in generated code")
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    # -------------------------
    # 6. FALLBACK SUGGESTION
    # -------------------------
    def suggest_fallback(self, test_result: Dict) -> Dict:
        """
        Suggest fallback options based on test failure
        
        Args:
            test_result: Result from run_tests()
            
        Returns:
            Dict with fallback recommendations
        """
        if test_result.get("passed"):
            return {"needs_fallback": False}
        
        error = test_result.get("error", "")
        error_type = test_result.get("error_type", "")
        
        fallback_suggestions = {
            "import_error": {
                "action": "Use template fallback",
                "reason": "Missing Python packages",
                "fix": "pip install " + (test_result.get("missing_module", "langchain")),
                "use_template": True
            },
            "general_error": {
                "action": "Use template fallback",
                "reason": error[:100],
                "use_template": True
            }
        }
        
        suggestion = fallback_suggestions.get(error_type, fallback_suggestions["general_error"])
        
        return {
            "needs_fallback": True,
            "suggestion": suggestion,
            "original_error": error
        }


# -------------------------
# QUICK TEST FUNCTION
# -------------------------

def test_tester():
    """Test the tester with a sample agent"""
    
    # Create a test agent file
    test_agent_code = '''
class Agent:
    def __init__(self):
        pass
    
    def run(self, user_input: str) -> str:
        return f"You said: {user_input}"

if __name__ == "__main__":
    agent = Agent()
    print("Agent ready")
'''
    
    os.makedirs("generated_agents", exist_ok=True)
    test_file = "generated_agents/test_agent.py"
    
    with open(test_file, "w") as f:
        f.write(test_agent_code)
    
    # Run tests
    tester = Stage4Tester()
    result = tester.run_tests(test_file)
    
    print("\n" + "="*50)
    print("TESTER SELF-TEST RESULTS")
    print("="*50)
    print(f"Passed: {result['passed']}")
    print(f"Results: {result['test_results']}")
    if result.get('warnings'):
        print(f"Warnings: {result['warnings']}")
    
    # Test fallback suggestion
    if not result['passed']:
        fallback = tester.suggest_fallback(result)
        print(f"\nFallback Suggestion: {fallback}")
    
    # Cleanup
    os.remove(test_file)
    
    return result


if __name__ == "__main__":
    test_tester()