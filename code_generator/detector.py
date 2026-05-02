# code_generator/detector.py

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ToolDetector:
    TOOL_PATTERNS = {
        "calculator": {
            "keywords": ["calculate", "math", "equation", "solve", "compute", "arithmetic"],
            "patterns": [r'\d+\s*[\+\-\*\/]\s*\d+', r'calculate\s+', r'what is\s+\d+'],
            "priority": 10
        },
        "search": {
            "keywords": ["search", "find", "lookup", "query", "web", "google", "bing"],
            "patterns": [r'search\s+for', r'find\s+information', r'look up'],
            "priority": 8
        },
        "retriever": {
            "keywords": ["pdf", "document", "file", "retrieve", "vector", "knowledge base"],
            "patterns": [r'pdf\s+qa', r'document\s+question', r'retrieve\s+from'],
            "priority": 9
        },
        "memory": {
            "keywords": ["remember", "memory", "context", "conversation", "history"],
            "patterns": [r'remember\s+my', r'conversation\s+history'],
            "priority": 7
        }
    }
    
    @classmethod
    def detect(cls, user_input: str, plan_tools: Optional[List[str]] = None) -> List[str]:
        detected = set()
        user_lower = user_input.lower()
        for tool, cfg in cls.TOOL_PATTERNS.items():
            if any(k in user_lower for k in cfg["keywords"]):
                detected.add(tool)
            elif any(re.search(p, user_lower, re.IGNORECASE) for p in cfg["patterns"]):
                detected.add(tool)
        if plan_tools:
            detected.update(plan_tools)
        logger.info(f"Detected tools: {detected}")
        return list(detected)