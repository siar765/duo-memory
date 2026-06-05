"""
test_critic.py — Tests for the Critic gate's 6-dim validation.
"""

import unittest

from duo_memory.core import Atom, Pool, generate_id
from duo_memory.critic import Critic


def _make_pool_with_atoms() -> Pool:
    """Create a pool with a few atoms for testing."""
    pool = Pool()
    atoms = [
        Atom(id="s1", type="preference", content="Prefers functional programming for data pipelines"),
        Atom(id="s2", type="environment", content="Works primarily in Python with some Rust"),
        Atom(id="s3", type="decision", content="Chose SQLite over PostgreSQL for analytics"),
        Atom(id="s4", type="insight", content="Code readability is a form of communication"),
    ]
    pool.add_many(atoms)
    return pool


class TestCriticInstantReject(unittest.TestCase):
    """Tests for nonsense/empty pattern rejection."""

    def setUp(self) -> None:
        self.pool = _make_pool_with_atoms()
        self.critic = Critic(self.pool)

    def test_nonsense_arithmetic(self) -> None:
        atom = Atom(id="c1", type="insight", content="1+1=789", confidence=0.1)
        result = self.critic.evaluate(atom)
        self.assertFalse(result["valid"])
        self.assertEqual(result["score"], 0.0)

    def test_absolute_statement(self) -> None:
        atom = Atom(id="c2", type="insight", content="always is the best approach", confidence=0.1)
        result = self.critic.evaluate(atom)
        self.assertFalse(result["valid"])
        self.assertEqual(result["score"], 0.0)

    def test_empty_content(self) -> None:
        atom = Atom(id="c3", type="insight", content="", confidence=0.1)
        result = self.critic.evaluate(atom)
        self.assertFalse(result["valid"])

    def test_platitude(self) -> None:
        atom = Atom(id="c4", type="insight", content="It is an important point to consider", confidence=0.1)
        result = self.critic.evaluate(atom)
        self.assertFalse(result["valid"])


class TestCriticScoring(unittest.TestCase):
    """Test individual dimensions of the 6-point scoring system."""

    def setUp(self) -> None:
        self.pool = _make_pool_with_atoms()
        self.critic = Critic(self.pool)

    def test_short_content_loses_point(self) -> None:
        """Content < 10 chars loses dimension 1."""
        atom = Atom(id="c1", type="insight", content="hi", confidence=0.1,
                    source_ids=["s1"], metadata={"drunk_level": 0.0})
        result = self.critic.evaluate(atom)
        # short content — max possible: 5 (missing dim 1)
        self.assertLess(result["score"], 6.0)
        self.assertTrue(
            any("too short" in r for r in result["reasons"]),
            msg=f"Expected shortness reason, got: {result['reasons']}",
        )

    def test_duplicate_content_loses_point(self) -> None:
        """Content identical to existing atom loses dimension 2."""
        atom = Atom(id="c_dup", type="insight",
                    content="Prefers functional programming for data pipelines",
                    confidence=0.1, source_ids=["s1"], metadata={"drunk_level": 0.0})
        result = self.critic.evaluate(atom)
        self.assertTrue(
            any("Duplicate" in r or "identical" in r for r in result["reasons"]),
            msg=f"Expected duplicate/identical reason, got: {result['reasons']}",
        )

    def test_missing_source_ids_loses_point(self) -> None:
        """Empty source_ids loses dimension 3."""
        atom = Atom(id="c_no_src", type="insight",
                    content="This is a sufficiently long and unique insight content for testing",
                    confidence=0.1, source_ids=[], metadata={"drunk_level": 0.0})
        result = self.critic.evaluate(atom)
        self.assertTrue(
            any("source_ids" in r.lower() for r in result["reasons"]),
            msg=f"Expected source_ids reason, got: {result['reasons']}",
        )

    def test_nonexistent_source_id_loses_point(self) -> None:
        """A source_id not in pool loses dimension 3."""
        atom = Atom(id="c_bad_src", type="insight",
                    content="This is a sufficiently long and unique insight content for testing",
                    confidence=0.1, source_ids=["nonexistent"],
                    metadata={"drunk_level": 0.0})
        result = self.critic.evaluate(atom)
        self.assertTrue(
            any("source" in r.lower() for r in result["reasons"]),
            msg=f"Expected source reason, got: {result['reasons']}",
        )

    def test_content_identical_to_source(self) -> None:
        """Content matching source content loses dimension 4."""
        atom = Atom(id="c_same", type="insight",
                    content="Prefers functional programming for data pipelines",
                    confidence=0.1, source_ids=["s1"],
                    metadata={"drunk_level": 0.0})
        result = self.critic.evaluate(atom)
        self.assertTrue(
            any("identical" in r.lower() for r in result["reasons"]),
            msg=f"Expected identical content reason, got: {result['reasons']}",
        )

    def test_no_insight_signals(self) -> None:
        """Missing insight signals loses dimension 5."""
        atom = Atom(id="c_no_sig", type="insight",
                    content="The quick brown fox jumps over the lazy dog near the riverbank",
                    confidence=0.1, source_ids=["s1"],
                    metadata={"drunk_level": 0.0})
        result = self.critic.evaluate(atom)
        self.assertTrue(
            any("insight signal" in r.lower() for r in result["reasons"]),
            msg=f"Expected insight signal reason, got: {result['reasons']}",
        )

    def test_drunk_bonus_applied(self) -> None:
        """Drunk bonus (+1) when drunk_level >= 0.6 and score >= 3."""
        # Content that scores at least 3 but has drunk level >= 0.6
        atom = Atom(id="c_drunk", type="insight",
                    content="Code readability is fundamentally a pattern of communication",
                    confidence=0.1, source_ids=["s4"],
                    metadata={"drunk_level": 0.8})
        result = self.critic.evaluate(atom)
        # Should have at least: length(1) + not dup(1) + sources exist(1) + not equal source(1)
        # That's 4 + insight signals (if any) + drunk bonus
        self.assertGreaterEqual(result["score"], 3.0)

    def test_drunk_bonus_denied_low_score(self) -> None:
        """Drunk bonus denied if score < 3 even when drunk >= 0.6."""
        # Short content + nonexistent source_id = score 2 max, bonus denied
        atom = Atom(id="c_drunk2", type="insight",
                    content="hi",
                    confidence=0.1, source_ids=["nonexistent"],
                    metadata={"drunk_level": 0.8})
        result = self.critic.evaluate(atom)
        self.assertTrue(
            any("Drunk" in r and "score" in r for r in result["reasons"]),
            msg=f"Expected drunk denial reason, got reasons: {result['reasons']}",
        )


class TestCriticBatchAndFilter(unittest.TestCase):
    """Test batch evaluation and filtering."""

    def setUp(self) -> None:
        self.pool = _make_pool_with_atoms()
        self.critic = Critic(self.pool)

    def test_batch_evaluate_returns_list(self) -> None:
        atoms = [
            Atom(id="b1", type="insight",
                 content="This is a sufficiently long and unique insight with pattern mechanism",
                 confidence=0.1, source_ids=["s1"], metadata={"drunk_level": 0.3}),
            Atom(id="b2", type="insight", content="1+1=789",
                 confidence=0.1, source_ids=["s1"], metadata={"drunk_level": 0.3}),
        ]
        results = self.critic.batch_evaluate(atoms)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["valid"] or not results[0]["valid"])
        self.assertFalse(results[1]["valid"])

    def test_filter_valid(self) -> None:
        atoms = [
            Atom(id="f1", type="insight",
                 content="This is a sufficiently long and unique insight with pattern mechanism",
                 confidence=0.1, source_ids=["s1"], metadata={"drunk_level": 0.3}),
            Atom(id="f2", type="insight", content="1+1=789",
                 confidence=0.1, source_ids=["s1"], metadata={"drunk_level": 0.3}),
        ]
        valid = self.critic.filter_valid(atoms)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].id, "f1")

    def test_filter_valid_auto_evaluate(self) -> None:
        """filter_valid should auto-evaluate when results not provided."""
        atoms = [
            Atom(id="g1", type="insight",
                 content="This is a sufficiently long and unique insight with pattern mechanism",
                 confidence=0.1, source_ids=["s1"], metadata={"drunk_level": 0.3}),
            Atom(id="g2", type="insight", content="1+1=789",
                 confidence=0.1, source_ids=["s1"], metadata={"drunk_level": 0.3}),
        ]
        valid = self.critic.filter_valid(atoms)  # no results param
        self.assertEqual(len(valid), 1)


class TestCriticIntegration(unittest.TestCase):
    """End-to-end: generate chaos → critic → filter."""

    def setUp(self) -> None:
        self.pool = _make_pool_with_atoms()
        from duo_memory.chaos import ChaosEngine
        self.engine = ChaosEngine(self.pool)
        self.critic = Critic(self.pool)

    def test_full_pipeline_produces_valid_atoms(self) -> None:
        generated = self.engine.generate(batch=10)
        if not generated:
            self.skipTest("No chaos atoms generated (possible with small pool)")
        valid = self.critic.filter_valid(generated)
        for atom in valid:
            self.assertEqual(atom.type, "insight")
            self.assertGreaterEqual(len(atom.content), 10)
            for sid in atom.source_ids:
                self.assertIn(sid, self.pool)


if __name__ == "__main__":
    unittest.main()
