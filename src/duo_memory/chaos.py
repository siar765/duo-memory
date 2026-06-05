"""
chaos.py — 7 drunk-but-logical chaos operations.

The ChaosEngine takes atoms from a Pool and applies controlled-randomness
operations to generate creative "insight" atoms. Each operation preserves
enough logical structure to be potentially meaningful while introducing
enough noise to discover novel connections.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Atom, Pool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beta_sample(alpha: float, beta: float) -> float:
    """Sample from a Beta distribution using the Python stdlib only.

    Uses the Bazett-Marsaglia transformation (rejection sampling).
    """
    # Based on algorithm BA (beta deviate) from Cheng 1978
    # Simplified: use the fact that Beta(a,b) ~ Gamma(a)/Gamma(a+b)
    # We'll use the Marsaglia-Tsang method for Gamma
    def _gamma_sample(shape: float) -> float:
        """Sample from Gamma(shape, 1) using Marsaglia-Tsang method (shape >= 1)."""
        if shape < 1:
            # Use shape + 1 then adjust
            g = _gamma_sample(shape + 1)
            return g * (random.random() ** (1.0 / shape))
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = _gauss_sample()
            v = 1.0 + c * x
            if v <= 0:
                continue
            v = v * v * v
            u = random.random()
            if u < 1.0 - 0.0331 * (x * x) * (x * x):
                return d * v
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    def _gauss_sample() -> float:
        """Box-Muller transform."""
        u1 = random.random()
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    g1 = _gamma_sample(alpha)
    g2 = _gamma_sample(beta)
    return g1 / (g1 + g2)


def _gauss_jitter(mean: float = 0.0, std: float = 0.1) -> float:
    """Add Gaussian noise and clamp to [0, 1]."""
    u1 = random.random()
    u2 = random.random()
    g = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return max(0.0, min(1.0, mean + g * std))


def _synonym_swap(text: str) -> str:
    """Swap a few words with crude synonyms from a tiny built-in map."""
    SYNONYMS = {
        "good": "beneficial", "bad": "harmful", "big": "substantial",
        "small": "minor", "fast": "rapid", "slow": "gradual",
        "high": "elevated", "low": "reduced", "new": "novel",
        "old": "established", "use": "leverage", "make": "produce",
        "get": "acquire", "find": "discover", "show": "demonstrate",
        "change": "shift", "help": "facilitate", "cause": "induce",
        "need": "require", "want": "prefer", "like": "appreciate",
        "think": "consider", "know": "understand", "see": "observe",
        "try": "attempt", "look": "examine", "ask": "inquire",
        "answer": "respond", "start": "initiate", "stop": "cease",
        "work": "function", "run": "execute", "move": "shift",
        "put": "position", "set": "configure", "keep": "maintain",
    }
    words = text.split()
    swaps = min(max(1, len(words) // 5), 3)
    for _ in range(swaps):
        idx = random.randrange(len(words))
        lower = words[idx].lower().strip(".,!?;:")
        if lower in SYNONYMS:
            replacement = SYNONYMS[lower]
            # Preserve case
            if words[idx][0].isupper():
                replacement = replacement.capitalize()
            words[idx] = replacement
    return " ".join(words)


def _common_words() -> List[str]:
    """Return a small list of common domain-agnostic words for semantic drift."""
    return [
        "pattern", "system", "process", "structure", "cycle", "flow",
        "network", "layer", "boundary", "threshold", "signal", "feedback",
        "balance", "tension", "rhythm", "scale", "phase", "state",
        "transition", "mechanism", "function", "resource", "constraint",
        "adaptation", "evolution", "emergence", "convergence", "divergence",
    ]


# ---------------------------------------------------------------------------
# ChaosEngine
# ---------------------------------------------------------------------------

class ChaosEngine:
    """Generates creative insight atoms through controlled random operations."""

    def __init__(self, pool: "Pool") -> None:
        self.pool = pool

    # ---- Drunkenness -----------------------------------------------------

    def get_drunkenness_level(self) -> float:
        """Return a drunkenness level in [0, 1] from Beta(0.5,0.5) + jitter.

        Beta(0.5,0.5) produces a U-shaped distribution — the engine is
        usually either quite sober or quite drunk, with occasional middle
        states. Gaussian jitter is added for unpredictability.
        """
        beta = _beta_sample(0.5, 0.5)
        return max(0.0, min(1.0, beta + _gauss_jitter(0.0, 0.08)))

    # ---- Operation selection ---------------------------------------------

    def pick_operation(self, level: float) -> str:
        """Pick an operation based on drunkenness level.

        Sober (0.0-0.4):    sober_rephrase, analogical_bridge
        Tipsy  (0.4-0.65):  cross_domain_transfer, category_blend
        Drunk  (0.65-0.85): polar_shift, semantic_drift
        Wasted(0.85-1.0):   extreme_analogy (and any of the above)
        """
        ops = {
            "sober_rephrase": (0.0, 0.4),
            "analogical_bridge": (0.0, 0.4),
            "cross_domain_transfer": (0.4, 0.65),
            "category_blend": (0.4, 0.65),
            "polar_shift": (0.65, 0.85),
            "semantic_drift": (0.65, 0.85),
            "extreme_analogy": (0.85, 1.0),
        }
        candidates = [name for name, (lo, hi) in ops.items() if lo <= level <= hi]
        if not candidates:
            # Fallback: widen to nearest
            candidates = list(ops.keys())
        return random.choice(candidates)

    # ---- 7 operations ----------------------------------------------------

    def sober_rephrase(self, atom: "Atom") -> Optional["Atom"]:
        """Minimal rephrase — swap synonyms, preserve meaning."""
        new_content = _synonym_swap(atom.content)
        if new_content == atom.content:
            return None  # no change possible
        return self._make_chaos_atom(
            content=new_content,
            sources=[atom.id],
            op_name="sober_rephrase",
        )

    def analogical_bridge(
        self, atom_a: "Atom", atom_b: "Atom", pool: "Pool"
    ) -> Optional["Atom"]:
        """Find structural similarity between domains.

        Produces "X is like Y because both ..." style insights.
        """
        # Extract key nouns from each
        words_a = [w for w in atom_a.content.split() if len(w) > 3][:5]
        words_b = [w for w in atom_b.content.split() if len(w) > 3][:5]

        if not words_a or not words_b:
            return None

        # Pick a representative word from each
        rep_a = random.choice(words_a).strip(".,!?;:")
        rep_b = random.choice(words_b).strip(".,!?;:")
        candidate = (
            f"There is a structural parallel between '{rep_a}' in "
            f"[{atom_a.type}] and '{rep_b}' in [{atom_b.type}]: "
            f"both serve as organizing constraints within their respective systems, "
            f"shaping outcomes not by direct force but by boundary definition."
        )
        return self._make_chaos_atom(
            content=candidate,
            sources=[atom_a.id, atom_b.id],
            op_name="analogical_bridge",
        )

    def cross_domain_transfer(
        self, pattern_atom: "Atom", target_atom: "Atom", pool: "Pool"
    ) -> Optional["Atom"]:
        """Apply a pattern from one domain to another.

        Takes the structural logic of *pattern_atom* and applies it to the
        domain of *target_atom*.
        """
        # Extract a verb-like pattern phrase from pattern_atom
        words_p = pattern_atom.content.split()
        if len(words_p) < 4:
            return None
        verb_candidates = [w for w in words_p if len(w) > 3][:3]
        if not verb_candidates:
            return None
        verb = random.choice(verb_candidates).strip(".,!?;:")

        # Extract a domain noun from target
        words_t = target_atom.content.split()
        noun_candidates = [w for w in words_t if len(w) > 3][:3]
        if not noun_candidates:
            return None
        noun = random.choice(noun_candidates).strip(".,!?;:")

        candidate = (
            f"The principle of '{verb}' from [{pattern_atom.type}] "
            f"can be transferred to the domain of '{noun}' in [{target_atom.type}]: "
            f"what operates as a dynamic driver in one context may act as "
            f"a stabilizing constraint in another."
        )
        return self._make_chaos_atom(
            content=candidate,
            sources=[pattern_atom.id, target_atom.id],
            op_name="cross_domain_transfer",
        )

    def category_blend(self, atom_a: "Atom", atom_b: "Atom") -> Optional["Atom"]:
        """Blend type + content across two atoms.

        Produces a hybrid that combines the structural role of one with
        the domain of the other.
        """
        types = [atom_a.type, atom_b.type]
        blended_type = random.choice(types)

        words_a = atom_a.content.split()
        words_b = atom_b.content.split()
        if not words_a or not words_b:
            return None

        # Take first half of a, second half of b
        mid_a = len(words_a) // 2
        mid_b = len(words_b) // 2
        blended = " ".join(words_a[:mid_a] + words_b[mid_b:])

        candidate = (
            f"[{blended_type}] blend: {blended} — "
            f"combining the framing of '{atom_a.type}' with the "
            f"context of '{atom_b.type}' reveals hidden compatibility."
        )
        return self._make_chaos_atom(
            content=candidate,
            sources=[atom_a.id, atom_b.id],
            op_name="category_blend",
        )

    def polar_shift(self, atom: "Atom") -> Optional["Atom"]:
        """Invert a fact productively.

        "What if the opposite is also valid under some conditions?"
        """
        content = atom.content
        # Simple polarity flipping heuristics
        negated = False
        for prefix in ["never ", "always ", "cannot ", "must ", "will "]:
            if prefix in content:
                content = content.replace(prefix, f"sometimes not ", 1)
                negated = True
                break

        if not negated:
            # Try flipping positive/negative framing
            if "increases" in content:
                content = content.replace("increases", "can also decrease", 1)
                negated = True
            elif "decreases" in content:
                content = content.replace("decreases", "can also increase", 1)
                negated = True
            elif "beneficial" in content:
                content = content.replace("beneficial", "conditionally detrimental", 1)
                negated = True

        if not negated:
            return None

        candidate = (
            f"Polar shift of [{atom.type}]: {content} — "
            f"the inverse frame reveals boundary conditions of the original."
        )
        return self._make_chaos_atom(
            content=candidate,
            sources=[atom.id],
            op_name="polar_shift",
        )

    def semantic_drift(self, atom: "Atom") -> Optional["Atom"]:
        """Shift one key word/attribute while keeping the logical skeleton.

        Replaces a meaningful noun with a random domain-agnostic concept.
        """
        words = atom.content.split()
        if len(words) < 4:
            return None

        # Find a plausible word to replace (longer noun-like words)
        candidates_idx = [
            i for i, w in enumerate(words)
            if len(w.strip(".,!?;:")) > 4 and w[0].islower()
        ]
        if not candidates_idx:
            return None

        idx = random.choice(candidates_idx)
        old_word = words[idx].strip(".,!?;:")
        new_word = random.choice(_common_words())
        words[idx] = words[idx].replace(old_word, new_word, 1)

        candidate = (
            f"Semantic drift ({old_word} -> {new_word}): "
            f"{' '.join(words)} — the logical skeleton holds despite "
            f"the lexical substitution."
        )
        return self._make_chaos_atom(
            content=candidate,
            sources=[atom.id],
            op_name="semantic_drift",
        )

    def extreme_analogy(
        self, atom_a: "Atom", atom_b: "Atom"
    ) -> Optional["Atom"]:
        """Far-fetched comparison that surprisingly works.

        Connects two atoms from entirely different domains through a
        shared abstract structure.
        """
        word_a = random.choice(
            [w for w in atom_a.content.split() if len(w) > 3] or atom_a.content.split()
        ).strip(".,!?;:")
        word_b = random.choice(
            [w for w in atom_b.content.split() if len(w) > 3] or atom_b.content.split()
        ).strip(".,!?;:")

        candidate = (
            f"Wild analogy: '{word_a}' ({atom_a.type}) is like "
            f"'{word_b}' ({atom_b.type}) — seemingly unrelated, yet both "
            f"embody the abstract principle of asymmetric constraint "
            f"distribution within a self-organizing system."
        )
        return self._make_chaos_atom(
            content=candidate,
            sources=[atom_a.id, atom_b.id],
            op_name="extreme_analogy",
        )

    # ---- Generator -------------------------------------------------------

    def generate(self, batch: int = 5) -> List["Atom"]:
        """Run a chaos cycle: random drunk level → pick op → apply → collect.

        Returns a list of newly-created insight-type atoms.
        """
        results: List["Atom"] = []
        pool_size = len(self.pool)
        if pool_size < 2:
            return results

        for _ in range(batch):
            level = self.get_drunkenness_level()
            op_name = self.pick_operation(level)

            atom: Optional["Atom"] = None

            # Operations needing 1 atom
            if op_name in ("sober_rephrase", "polar_shift", "semantic_drift"):
                a = self.pool.random(1)
                if not a:
                    continue
                if op_name == "sober_rephrase":
                    atom = self.sober_rephrase(a[0])
                elif op_name == "polar_shift":
                    atom = self.polar_shift(a[0])
                elif op_name == "semantic_drift":
                    atom = self.semantic_drift(a[0])

            # Operations needing 2 atoms
            elif op_name in ("analogical_bridge", "category_blend", "extreme_analogy"):
                pair = self.pool.random(2)
                if len(pair) < 2:
                    continue
                if op_name == "analogical_bridge":
                    atom = self.analogical_bridge(pair[0], pair[1], self.pool)
                elif op_name == "category_blend":
                    atom = self.category_blend(pair[0], pair[1])
                elif op_name == "extreme_analogy":
                    atom = self.extreme_analogy(pair[0], pair[1])

            elif op_name == "cross_domain_transfer":
                pair = self.pool.random(2)
                if len(pair) < 2:
                    continue
                atom = self.cross_domain_transfer(pair[0], pair[1], self.pool)

            if atom is not None:
                results.append(atom)

        return results

    # ---- Internal --------------------------------------------------------

    def _make_chaos_atom(
        self,
        content: str,
        sources: List[str],
        op_name: str,
    ) -> "Atom":
        """Create a chaos-generated insight atom with proper metadata."""
        from .core import Atom, generate_id

        return Atom(
            id=generate_id(),
            type="insight",
            content=content,
            confidence=round(random.uniform(0.05, 0.5), 4),
            evidence=f"Generated by chaos operation '{op_name}'",
            source_ids=sources,
            metadata={
                "chaos_op": op_name,
                "drunk_level": round(self.get_drunkenness_level(), 4),
            },
        )
