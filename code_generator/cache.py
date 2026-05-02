# code_generator/cache.py

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CodeCache:
    """Cache generated code to avoid redundant LLM calls"""
    
    def __init__(self, cache_file: str = "generated_agents/code_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached generations")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
    
    def _save(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _make_key(self, user_request: str, agent_type: str, tools: List[str], constraints: Dict) -> str:
        content = f"{user_request}|{agent_type}|{sorted(tools)}|{json.dumps(constraints, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, user_request: str, agent_type: str, tools: List[str], constraints: Dict) -> Optional[Dict]:
        key = self._make_key(user_request, agent_type, tools, constraints)
        if key in self.cache:
            logger.info(f"Cache HIT for request: {user_request[:50]}...")
            return self.cache[key]
        return None
    
    def put(self, user_request: str, agent_type: str, tools: List[str], constraints: Dict, result: Dict):
        key = self._make_key(user_request, agent_type, tools, constraints)
        self.cache[key] = {
            "code": result.get("code"),
            "generation_method": result.get("generation_method"),
            "timestamp": result.get("timestamp", ""),
            "quality_score": result.get("quality_score")
        }
        self._save()
        logger.info(f"Cached result for request: {user_request[:50]}...")