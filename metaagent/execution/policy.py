"""Static policy check for generated code.

THIS IS NOT A SANDBOX. It is a cheap pre-filter that rejects obviously
unwanted constructs before code is run, and it produces feedback the LLM can
act on during self-correction. Determined code can evade any AST denylist
(e.g. via attribute tricks or third-party modules). Containment is the job of
execution/runner.py, and full isolation (container, restricted filesystem,
resource limits) is planned for the deployment phase.
"""

import ast
from dataclasses import dataclass, field
from typing import List

FORBIDDEN_CALLS = {
    "__import__", "exec", "eval", "compile", "globals", "getattr", "setattr", "delattr",
    "breakpoint", "vars",
}
FORBIDDEN_ATTRIBUTES = {"__code__", "__globals__", "__builtins__", "__subclasses__", "environ"}
FORBIDDEN_MODULES = {
    "subprocess", "socket", "pickle", "marshal", "multiprocessing", "threading", "_thread",
    "ctypes", "winreg", "pdb", "shutil", "importlib", "builtins", "pty", "signal",
    "google.generativeai",
}
ALLOWED_OS_FUNCTIONS = {
    "getenv", "path", "listdir", "mkdir", "makedirs", "unlink",
    "exists", "isfile", "isdir", "join", "basename", "dirname",
}


@dataclass
class PolicyResult:
    passed: bool
    violations: List[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        return "Policy check passed" if self.passed else "; ".join(self.violations)


def _module_root(name: str) -> str:
    return name.split(".")[0] if name else ""


def _is_forbidden_module(name: str) -> bool:
    return name in FORBIDDEN_MODULES or _module_root(name) in FORBIDDEN_MODULES


def check_code(code: str) -> PolicyResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return PolicyResult(False, [f"Syntax error at line {e.lineno}: {e.msg}"])

    violations: List[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                violations.append(f"line {line}: forbidden call {func.id}()")
            elif isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "os" \
                        and func.attr not in ALLOWED_OS_FUNCTIONS:
                    violations.append(f"line {line}: forbidden os.{func.attr}()")
                elif func.attr in FORBIDDEN_CALLS:
                    violations.append(f"line {line}: forbidden call .{func.attr}()")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            violations.append(f"line {line}: forbidden attribute .{node.attr}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    violations.append(f"line {line}: forbidden import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden_module(module):
                violations.append(f"line {line}: forbidden import from {module}")
            elif module == "os":
                for alias in node.names:
                    if alias.name == "*" or alias.name not in ALLOWED_OS_FUNCTIONS:
                        violations.append(f"line {line}: forbidden import os.{alias.name}")
            elif module == "google" and any(a.name == "generativeai" for a in node.names):
                violations.append(f"line {line}: forbidden import google.generativeai")

    return PolicyResult(not violations, violations)
