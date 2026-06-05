"""critic.py — Critic gate with 6-dimensional validation.

Evaluates chaos-generated atoms against a set of quality criteria.
Filters out nonsense, empty talk, duplicates (exact + near-dupe),
and ungrounded claims.
"""

from __future__ import annotations

import math
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
    "本质上",
    "就像", "模式", "类比", "反过来",
    "实际上是", "其本质是",
    "analogy", "essentially", "fundamentally",
    "pattern", "mechanism", "structure",
    "parallel", "resonates", "mirrors",
    "emerges", "underlying", "architecture",
]

NEAR_DUPE_THRESHOLD = 0.85  # TF-IDF cosine similarity above this = near-duplicate


# ---------------------------------------------------------------------------
# TF-IDF helpers (pure Python, no deps)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Split text into lowercase tokens on non-word characters."""
    return [t.lower() for t in re.split(r"\W+", text) if t]


def _tfidf_cosine_similarity(a_text: str, b_texts: List[str]) -> float:
    """Max TF-IDF cosine similarity between *a_text* and any text in *b_texts*.

    Pure Python, builds vocabulary on the fly from all input texts.
    Returns 0.0 if no valid comparison can be made.
    """
    all_texts = [a_text] + list(b_texts) if b_texts else [a_text]
    all_tokens = [_tokenise(t) for t in all_texts]

    # Build vocabulary
    vocab: Dict[str, int] = {}
    for tokens in all_tokens:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    if not vocab:
        return 0.0

    n_docs = len(all_tokens)
    # IDF
    idf: Dict[str, float] = {}
    for term in vocab:
        df = sum(1 for tokens in all_tokens if term in tokens)
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

    # TF-IDF vectors
    vecs = []
    for tokens in all_tokens:
        vec = [0.0] * len(vocab)
        for t in tokens:
            if t in vocab:
                vec[vocab[t]] += 1.0  # TF
        # Apply IDF
        for term, idx in vocab.items():
            if vec[idx] > 0:
                vec[idx] *= idf[term]
        vecs.append(vec)

    # L2 normalize
    norms = []
    for vec in vecs:
        norm = math.sqrt(sum(v * v for v in vec))
        norms.append(norm if norm > 0 else 1.0)

    # Cosine similarity between first vector (a_text) and rest
    a_vec = vecs[0]
    a_norm = norms[0]

    max_sim = 0.0
    for i in range(1, len(vecs)):
        dot = sum(a_vec[j] * vecs[i][j] for j in range(len(vocab)))
        sim = dot / (a_norm * norms[i])
        max_sim = max(max_sim, sim)

    return max_sim


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

        # --- Dimension 2: Not a duplicate (exact or near) of any pool atom ---
        exact_dup = False
        near_dup_score = 0.0
        pool_contents = [a.content for a in self.pool if a.id not in atom.source_ids]

        for existing in self.pool:
            if existing.id not in atom.source_ids and existing.content == atom.content:
                exact_dup = True
                break

        if exact_dup:
            reasons.append("Duplicate content — identical to an existing atom")
        else:
            # Near-duplicate check via TF-IDF cosine similarity
            if pool_contents:
                near_dup_score = _tfidf_cosine_similarity(atom.content, pool_contents)
            if near_dup_score >= NEAR_DUPE_THRESHOLD:
                score += 0.5
                reasons.append(
                    f"Near-duplicate (TF-IDF sim={near_dup_score:.2f} >= {NEAR_DUPE_THRESHOLD})"
                )
            else:
                score += 1.0

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
