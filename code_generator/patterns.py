
import os
import json
import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Optional embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GenerationPattern:
    pattern_id: str
    keywords: List[str]
    embedding: Optional[List[float]]
    agent_type: str
    tools: List[str]
    generation_method: str
    success_count: int
    failure_count: int
    last_used: str
    
    @property
    def confidence(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    @property
    def should_use_llm(self) -> bool:
        if self.failure_count >= 3:
            return False
        if self.confidence > 0.7:
            return True
        return None


class PatternDatabase:
    def __init__(self, db_path: str = "generated_agents/pattern_db.json"):
        self.db_path = db_path
        self.patterns: Dict[str, GenerationPattern] = {}
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded")
            except Exception as e:
                logger.warning(f"Embedding model failed: {e}")
        self._load()
    
    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for pid, pdata in data.items():
                        self.patterns[pid] = GenerationPattern(**pdata)
                logger.info(f"Loaded {len(self.patterns)} patterns")
            except Exception as e:
                logger.error(f"Load failed: {e}")
    
    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            data = {pid: asdict(p) for pid, p in self.patterns.items()}
            json.dump(data, f, indent=2)
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        if self.encoder:
            try:
                return self.encoder.encode(text).tolist()
            except:
                pass
        return None
    
    def _cosine_similarity(self, emb1, emb2) -> float:
        if not emb1 or not emb2 or not EMBEDDINGS_AVAILABLE:
            return 0.0
        vec1, vec2 = np.array(emb1), np.array(emb2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def find_matching_pattern(self, user_request: str, agent_type: str, tools: List[str]) -> Optional[Tuple[GenerationPattern, float]]:
        best_match, best_score = None, 0.0
        query_embed = self._get_embedding(user_request.lower())
        for pattern in self.patterns.values():
            if pattern.agent_type != agent_type:
                continue
            if query_embed and pattern.embedding:
                sim = self._cosine_similarity(query_embed, pattern.embedding)
            else:
                kw1 = set(re.findall(r'\b[a-z]{3,}\b', user_request.lower()))
                kw2 = set(pattern.keywords)
                sim = len(kw1 & kw2) / max(len(kw1), len(kw2)) if kw1 or kw2 else 0.0
            tool_overlap = len(set(tools) & set(pattern.tools))
            tool_score = tool_overlap / max(len(tools), len(pattern.tools)) if tools or pattern.tools else 1.0
            final = sim * 0.7 + tool_score * 0.3
            if final > best_score and final > 0.5:
                best_score, best_match = final, pattern
        return (best_match, best_score) if best_match else None
    
    def record_result(self, user_request: str, agent_type: str, tools: List[str],
                      generation_method: str, success: bool, code_quality: float = 1.0):
        pid = hashlib.md5(f"{user_request}_{agent_type}_{sorted(tools)}".encode()).hexdigest()[:12]
        if pid in self.patterns:
            p = self.patterns[pid]
            if success:
                p.success_count += 1
            else:
                p.failure_count += 1
            p.last_used = datetime.now().isoformat()
        else:
            self.patterns[pid] = GenerationPattern(
                pattern_id=pid,
                keywords=re.findall(r'\b[a-z]{3,}\b', user_request.lower())[:5],
                embedding=self._get_embedding(user_request.lower()),
                agent_type=agent_type,
                tools=tools,
                generation_method=generation_method if success else "template",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                last_used=datetime.now().isoformat()
            )
        self._save()