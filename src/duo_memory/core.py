"""
core.py — Atom dataclass, Pool manager, generate_id helper.

Defines the fundamental data structure (Atom) and the in-memory container (Pool)
that stores, searches, persists, and inspects atoms.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATOM_TYPES = frozenset({
    "preference", "environment", "decision", "rejection",
    "pattern", "relationship", "insight",
})

ID_CHARS = "0123456789abcdef"
ID_LENGTH = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_id() -> str:
    """Return an 8-hex-char random identifier."""
    return "".join(random.choice(ID_CHARS) for _ in range(ID_LENGTH))


def _validate_atom_type(t: str) -> None:
    if t not in ATOM_TYPES:
        raise ValueError(
            f"Unknown atom type {t!r}. Must be one of: {sorted(ATOM_TYPES)}"
        )


def _default_timestamp() -> float:
    """Return current Unix timestamp."""
    import time
    return time.time()


def _simple_tfidf_search(
    atoms: List["Atom"],
    query: str,
    top_k: int = 10,
) -> List["Atom"]:
    """
    Pure-Python TF-IDF search on atom content.

    Tokenises both query and atom content by splitting on non-word characters.
    Returns up to *top_k* atoms sorted by descending cosine similarity.
    """
    # --- tokeniser ---
    def _tokenise(text: str) -> List[str]:
        return [t.lower() for t in re.split(r"\W+", text) if t]

    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    # Build vocabulary across *atoms*
    doc_tokens: List[List[str]] = [_tokenise(a.content) for a in atoms]
    vocab: Dict[str, int] = {}
    for dt in doc_tokens:
        for t in dt:
            if t not in vocab:
                vocab[t] = len(vocab)

    # IDF
    n_docs = len(atoms)
    idf: Dict[str, float] = {}
    for term in vocab:
        df = sum(1 for dt in doc_tokens if term in dt)
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

    # Query TF-IDF vector
    q_vec = _tf_vector(query_tokens, vocab)
    q_norm = _l2_norm(q_vec)

    scored: List[Tuple[float, int]] = []
    for i, dt in enumerate(doc_tokens):
        d_vec = _tf_vector(dt, vocab)
        # apply IDF
        for term_idx in vocab.values():
            q_vec[term_idx] *= idf.get(list(vocab.keys())[term_idx], 1.0) if False else 1.0
        # Rebuild properly
        q_vec2 = [0.0] * len(vocab)
        d_vec2 = [0.0] * len(vocab)
        for term in set(query_tokens):
            if term in vocab:
                idx = vocab[term]
                q_vec2[idx] = query_tokens.count(term) * idf[term]
        for term in set(dt):
            if term in vocab:
                idx = vocab[term]
                d_vec2[idx] = dt.count(term) * idf[term]

        q_norm2 = _l2_norm(q_vec2)
        d_norm = _l2_norm(d_vec2)
        if q_norm2 == 0.0 or d_norm == 0.0:
            continue
        dot = sum(a * b for a, b in zip(q_vec2, d_vec2))
        scored.append((dot / (q_norm2 * d_norm), i))

    scored.sort(key=lambda x: -x[0])
    return [atoms[i] for _, i in scored[:top_k]]


def _tf_vector(tokens: List[str], vocab: Dict[str, int]) -> List[float]:
    vec = [0.0] * len(vocab)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    return vec


def _l2_norm(vec: List[float]) -> float:
    s = sum(v * v for v in vec)
    return math.sqrt(s)


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    """A single atomic fact in the duo-memory system."""

    id: str
    type: str  # one of ATOM_TYPES
    content: str
    confidence: float = 1.0
    evidence: str = ""
    source_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=_default_timestamp)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_atom_type(self.type)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")
        # Ensure mutable defaults
        if isinstance(self.source_ids, list):
            self.source_ids = list(self.source_ids)
        else:
            self.source_ids = []
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Atom":
        return cls(**d)


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class Pool:
    """In-memory container for Atom objects with search, stats & persistence."""

    def __init__(self, atoms: Optional[List[Atom]] = None) -> None:
        self._atoms: Dict[str, Atom] = {}
        if atoms:
            for a in atoms:
                self._atoms[a.id] = a

    # ---- CRUD ------------------------------------------------------------

    def add(self, atom: Atom) -> None:
        """Add a single atom. Overwrites if id already exists."""
        self._atoms[atom.id] = atom

    def add_many(self, atoms: List[Atom]) -> None:
        """Add multiple atoms in bulk."""
        for a in atoms:
            self._atoms[a.id] = a

    def remove(self, atom_id: str) -> None:
        """Remove an atom by id. KeyError if missing."""
        del self._atoms[atom_id]

    def get(self, atom_id: str) -> Atom:
        """Retrieve an atom by id. KeyError if missing."""
        return self._atoms[atom_id]

    # ---- Query -----------------------------------------------------------

    def search(self, query: str) -> List[Atom]:
        """Simple keyword / TF-IDF search on content."""
        atoms = list(self._atoms.values())
        if not query.strip():
            return atoms
        return _simple_tfidf_search(atoms, query)

    def random(self, n: int) -> List[Atom]:
        """Random sample of *n* atoms (may return fewer if pool is smaller)."""
        if n <= 0:
            return []
        return random.sample(list(self._atoms.values()), min(n, len(self._atoms)))

    def weighted_random(self, n: int) -> List[Atom]:
        """Random sample weighted by confidence — high-confidence atoms more likely."""
        atoms = list(self._atoms.values())
        if n <= 0 or not atoms:
            return []
        weights = [max(a.confidence, 0.01) for a in atoms]
        chosen = random.choices(atoms, weights=weights, k=min(n, len(atoms)))
        # Deduplicate while preserving weighting intent
        seen: set = set()
        result: list = []
        for a in chosen:
            if a.id not in seen:
                seen.add(a.id)
                result.append(a)
                if len(result) == n:
                    break
        return result or atoms[:min(n, len(atoms))]

    def by_type(self, atom_type: str) -> List[Atom]:
        """Return all atoms of the given type."""
        _validate_atom_type(atom_type)
        return [a for a in self._atoms.values() if a.type == atom_type]

    def stats(self) -> Dict:
        """Pool statistics: count, type distribution, average confidence."""
        atoms = list(self._atoms.values())
        total = len(atoms)
        if total == 0:
            return {"count": 0, "types": {}, "avg_confidence": 0.0}

        type_dist: Dict[str, int] = {}
        conf_sum = 0.0
        for a in atoms:
            type_dist[a.type] = type_dist.get(a.type, 0) + 1
            conf_sum += a.confidence

        return {
            "count": total,
            "types": type_dist,
            "avg_confidence": round(conf_sum / total, 4),
        }

    # ---- Persistence -----------------------------------------------------

    def to_jsonl(self, path: str) -> None:
        """Write all atoms as JSONL to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for atom in self._atoms.values():
                fh.write(json.dumps(atom.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "Pool":
        """Load atoms from a JSONL file and return a new Pool."""
        atoms: List[Atom] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    atoms.append(Atom.from_dict(json.loads(line)))
        return cls(atoms)

    @classmethod
    def load_from_agent_memory(cls, path: str) -> "Pool":
        """Load atoms from an agent-memory atoms directory.

        Merges all *.jsonl files found in *path* and tags each atom's
        metadata with ``source_repo="agent-memory"``.

        Args:
            path: Directory path containing *.jsonl atom files.

        Returns:
            A new Pool with all loaded atoms.
        """
        pool = cls()
        if not os.path.isdir(path):
            return pool
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".jsonl"):
                fpath = os.path.join(path, fname)
                try:
                    chunk = cls.from_jsonl(fpath)
                    for atom in chunk:
                        if not isinstance(atom.metadata, dict):
                            atom.metadata = {}
                        atom.metadata["source_repo"] = "agent-memory"
                    pool.add_many(list(chunk))
                except Exception:
                    continue  # skip corrupt files
        return pool

    # ---- Dunder ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._atoms)

    def __iter__(self) -> Iterator[Atom]:
        return iter(self._atoms.values())

    def __contains__(self, atom_id: str) -> bool:
        return atom_id in self._atoms
