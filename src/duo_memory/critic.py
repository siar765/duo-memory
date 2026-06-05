"""
critic.py — Critic gate with 6-dimensional validation.

Evaluates chaos-generated atoms against a set of quality criteria.
Filters out nonsense, empty talk, duplicates, and ungrounded claims.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Atom, Pool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NONSENSE_PATTERNS: List[str] = [
    r"\d+\s*\+\s*\d+\s*=\s*\d+",       # arithmetic nonsense
    r"1\s*\+\s*1\s*=\s*789",            # specific nonsense pattern
    r"\b(always|never|everyone|nobody|everything|nothing)\s+(is|will|does)",
    r"\b(绝对|必然|永远|所有)\b",         # absolute statements (Chinese)
    r"\b(impossible|infallible|perfect)\b\s+\b(always|never)\b",
]

EMPTY_PATTERNS: List[str] = [
    r"^\s*$",
    r"^\s*(it|this|that|there|here)\s+(is|was|are|were)\s+(a|an|the)\s+\w+\s*\.?\s*$",
    r"\b(套话|空话|废话|众所周知|毫无疑问)\b",
    r"^\s*(as we all know|it goes without saying|needless to say)",
    r"\b(interesting|noteworthy|important)\s+(point|thing|aspect|observation)\b",
]

INSIGHT_SIGNALS: List[str] = [
    "本质上", "本质上",  # same word in list for weight
    "就像", "模式", "类比", "反过来",
    "实际上是", "其本质是",
    "analogy", "essentially", "fundamentally",
    "pattern", "mechanism", "structure",
    "parallel", "resonates", "mirrors",
    "emerges", "underlying", "architecture",
]


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic:
    """Validates atoms against quality criteria and filters out low-quality ones."""

    def __init__(self, pool: "Pool") -> None:
        self.pool = pool
        self._nonsense_re = [re.compile(p, re.IGNORECASE) for p in NONSENSE_PATTERNS]
        self._empty_re = [re.compile(p, re.IGNORECASE) for p in EMPTY_PATTERNS]

    # ---- Single evaluation -----------------------------------------------

    def evaluate(self, atom: "Atom") -> Dict:
        """Run the 6-dim critic gate on a single atom.

        Returns:
            valid: bool — passed the gate
            score: float — 0-6
            reasons: list[str] — deductions recorded
        """
        reasons: List[str] = []
        score = 0.0

        # --- Instant reject checks ---
        for pat in self._nonsense_re:
            if pat.search(atom.content):
                return {
                    "valid": False,
                    "score": 0.0,
                    "reasons": [f"Nonsense pattern matched: {pat.pattern}"],
                }

        for pat in self._empty_re:
            if pat.search(atom.content):
                return {
                    "valid": False,
                    "score": 0.0,
                    "reasons": [f"Empty/platitude pattern matched: {pat.pattern}"],
                }

        # --- Dimension 1: Content length >= 10 chars ---
        if len(atom.content) >= 10:
            score += 1.0
        else:
            reasons.append(f"Content too short ({len(atom.content)} chars < 10)")

        # --- Dimension 2: Not a duplicate of any pool atom ---
        is_dup = False
        for existing in self.pool:
            if existing.content == atom.content:
                is_dup = True
                break
        if not is_dup:
            score += 1.0
        else:
            reasons.append("Duplicate content — identical to an existing atom")

        # --- Dimension 3: All source_ids exist in pool ---
        all_sources_exist = True
        for sid in atom.source_ids:
            try:
                self.pool.get(sid)
            except KeyError:
                all_sources_exist = False
                break
        if all_sources_exist and atom.source_ids:
            score += 1.0
        else:
            if not atom.source_ids:
                reasons.append("No source_ids provided")
            else:
                reasons.append("Some source_ids do not exist in pool")

        # --- Dimension 4: Content not identical to any source ---
        content_unique = True
        for sid in atom.source_ids:
            try:
                src = self.pool.get(sid)
                if src.content == atom.content:
                    content_unique = False
                    break
            except KeyError:
                continue
        if content_unique:
            score += 1.0
        else:
            reasons.append("Content identical to source atom content")

        # --- Dimension 5: Insight signals (>= 2 keywords) ---
        match_count = sum(
            1 for signal in INSIGHT_SIGNALS if signal.lower() in atom.content.lower()
        )
        if match_count >= 2:
            score += 1.0
        elif match_count == 1:
            reasons.append(f"Only 1 insight signal found (need >= 2)")
        else:
            reasons.append("No insight signals found")

        # --- Dimension 6: Drunk bonus ---
        drunk_level = atom.metadata.get("drunk_level", 0.0)
        if isinstance(drunk_level, (int, float)) and drunk_level >= 0.6 and score >= 3.0:
            score += 1.0
        elif isinstance(drunk_level, (int, float)) and drunk_level >= 0.6:
            reasons.append(
                f"Drunk level ({drunk_level:.2f}) >= 0.6 but score ({score:.1f}) < 3"
            )

        valid = score >= 4.0

        return {
            "valid": valid,
            "score": round(score, 2),
            "reasons": reasons,
        }

    # ---- Batch evaluation ------------------------------------------------

    def batch_evaluate(self, atoms: List["Atom"]) -> List[Dict]:
        """Evaluate multiple atoms and return a list of result dicts."""
        return [self.evaluate(a) for a in atoms]

    def filter_valid(
        self,
        atoms: List["Atom"],
        results: Optional[List[Dict]] = None,
    ) -> List["Atom"]:
        """Return only atoms that passed evaluation.

        If *results* is None, atoms are evaluated first. Otherwise
        atoms and results are zipped by index.
        """
        if results is None:
            results = self.batch_evaluate(atoms)
        return [a for a, r in zip(atoms, results) if r.get("valid", False)]
