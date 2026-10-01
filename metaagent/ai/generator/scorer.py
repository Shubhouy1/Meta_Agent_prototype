import re

class CodeQualityScorer:
    @staticmethod
    def score(code: str) -> float:
        """Return quality score between 0 and 1"""
        score = 1.0
        lines = code.split('\n')
        
        # Penalize too few lines
        if len(lines) < 20:
            score -= 0.2
        # Penalize no docstrings
        if '"""' not in code and "'''" not in code:
            score -= 0.15
        # Penalize no error handling
        if 'try' not in code or 'except' not in code:
            score -= 0.15
        # Penalize no main guard
        if "if __name__" not in code:
            score -= 0.1
        # Bonus for type hints
        if ': str' in code or '-> str' in code:
            score += 0.05
        # Bonus for reasonable length
        if 50 <= len(lines) <= 200:
            score += 0.05
        return max(0.5, min(1.0, score))