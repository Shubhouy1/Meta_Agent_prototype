# code_generator/sandbox.py

import ast
import subprocess
import tempfile
import os
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class AdvancedSandbox:
    """AST-based security scanner + subprocess validation"""
    
    # Forbidden function calls (these are dangerous anywhere)
    # Removed 'input' - it's safe for interactive demos
    FORBIDDEN_ATTRIBUTES = {
        '__import__', 'exec', 'eval', 'compile',
        'globals', 'getattr', 'setattr', 'delattr',
        '__code__', '__globals__', '__builtins__',
        'breakpoint'
    }
    
    # Forbidden modules (importing these is dangerous)
    FORBIDDEN_MODULES = {
        'subprocess', 'socket', 'pickle',
        'multiprocessing', 'threading', '_thread',
        'ctypes', 'winreg', 'pdb'
    }
    
    # Allowed os functions (os is allowed, but specific dangerous functions are blocked)
    ALLOWED_OS_FUNCTIONS = {
        'getenv', 'path', 'listdir', 'mkdir', 'makedirs',
        'exists', 'isfile', 'isdir', 'join', 'basename', 'dirname'
    }
    
    @classmethod
    def _check_ast(cls, code: str) -> Tuple[bool, str]:
        """Analyze AST for dangerous patterns"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        
        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Block dangerous functions
                    if node.func.id in cls.FORBIDDEN_ATTRIBUTES:
                        return False, f"Forbidden function call: {node.func.id}()"
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous os operations
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                        if node.func.attr not in cls.ALLOWED_OS_FUNCTIONS:
                            return False, f"Dangerous os operation: os.{node.func.attr}()"
                    
                    # Block other dangerous attributes
                    if node.func.attr in cls.FORBIDDEN_ATTRIBUTES:
                        return False, f"Forbidden attribute: {node.func.attr}"
            
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in cls.FORBIDDEN_MODULES:
                        return False, f"Dangerous import: {alias.name}"
                    # Allow 'os' but with restrictions (handled above)
                    if alias.name == 'os':
                        # os is allowed, but will be checked at function call level
                        pass
            
            if isinstance(node, ast.ImportFrom):
                if node.module in cls.FORBIDDEN_MODULES:
                    return False, f"Dangerous import from: {node.module}"
                # Allow 'os' imports
                if node.module == 'os':
                    for alias in node.names:
                        if alias.name not in cls.ALLOWED_OS_FUNCTIONS and alias.name != '*':
                            if alias.name not in cls.FORBIDDEN_ATTRIBUTES:
                                # Warn but don't block for unknown os functions
                                logger.warning(f"Unknown os function imported: {alias.name}")
        
        return True, "AST check passed"
    
    @classmethod
    def validate(cls, code: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Validate code for safety and syntax
        
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # AST security check
        ast_ok, ast_msg = cls._check_ast(code)
        if not ast_ok:
            logger.warning(f"AST check failed: {ast_msg}")
            return False, ast_msg
        
        # Subprocess compilation check (syntax validation)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tmp = f.name
            
            result = subprocess.run(
                ['python', '-m', 'py_compile', tmp],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown compilation error"
                return False, f"Compilation error: {error_msg[:200]}"
            
            logger.info("Code validation passed")
            return True, "Validation passed"
            
        except subprocess.TimeoutExpired:
            return False, f"Validation timeout ({timeout}s)"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        finally:
            # Clean up temp file
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")