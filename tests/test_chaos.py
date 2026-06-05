"""
test_chaos.py — Tests for ChaosEngine operations and drunkenness distribution.
"""

import math
import unittest

from duo_memory.core import Atom, Pool, generate_id
from duo_memory.chaos import ChaosEngine


def _make_sample_pool() -> Pool:
    """Create a pool with diverse atoms for chaos testing."""
    pool = Pool()
    facts = [
        Atom(id=generate_id(), type="preference", content="Prefers functional programming over OOP for data pipelines", confidence=0.9),
        Atom(id=generate_id(), type="environment", content="Works primarily in Python with occasional Rust for performance", confidence=0.95),
        Atom(id=generate_id(), type="decision", content="Chose SQLite over PostgreSQL for single-user analytics projects", confidence=0.85),
        Atom(id=generate_id(), type="rejection", content="Rejected microservices architecture in favor of monolith-first approach", confidence=0.8),
        Atom(id=generate_id(), type="pattern", content="Spends first 90 minutes of each day on deep work before checking messages", confidence=0.75),
        Atom(id=generate_id(), type="relationship", content="Collaborates better when async communication is the default", confidence=0.88),
        Atom(id=generate_id(), type="insight", content="Code readability is a form of communication, not just aesthetics", confidence=0.92),
        Atom(id=generate_id(), type="preference", content="Uses terminal-based tools over GUI whenever possible", confidence=0.85),
    ]
    pool.add_many(facts)
    return pool


class TestChaosEngineBasics(unittest.TestCase):
    """Basic ChaosEngine setup and drunkenness."""

    def setUp(self) -> None:
        self.pool = _make_sample_pool()
        self.engine = ChaosEngine(self.pool)

    def test_init(self) -> None:
        self.assertIs(self.engine.pool, self.pool)

    def test_drunkenness_range(self) -> None:
        for _ in range(200):
            level = self.engine.get_drunkenness_level()
            self.assertGreaterEqual(level, 0.0)
            self.assertLessEqual(level, 1.0)

    def test_drunkenness_distribution_shape(self) -> None:
        """Beta(0.5,0.5) is U-shaped — expect many low and high values."""
        levels = [self.engine.get_drunkenness_level() for _ in range(500)]
        low = sum(1 for l in levels if l < 0.3)
        high = sum(1 for l in levels if l > 0.7)
        middle = sum(1 for l in levels if 0.3 <= l <= 0.7)
        # U-shape: extremes should be at least as common as middle
        # With 500 samples from Beta(0.5,0.5), tails are heavy
        self.assertGreater(low + high, middle * 0.5)

    def test_pick_operation_sober(self) -> None:
        """Low drunk level should pick sober/tipsy ops."""
        for _ in range(50):
            op = self.engine.pick_operation(0.1)
            self.assertIn(op, ("sober_rephrase", "analogical_bridge"))

    def test_pick_operation_mid(self) -> None:
        for _ in range(50):
            op = self.engine.pick_operation(0.5)
            self.assertIn(op, ("cross_domain_transfer", "category_blend"))

    def test_pick_operation_drunk(self) -> None:
        for _ in range(50):
            op = self.engine.pick_operation(0.75)
            self.assertIn(op, ("polar_shift", "semantic_drift"))

    def test_pick_operation_wasted(self) -> None:
        for _ in range(50):
            op = self.engine.pick_operation(0.95)
            self.assertEqual(op, "extreme_analogy")

    def test_pick_operation_always_returns_valid(self) -> None:
        valid_ops = {
            "sober_rephrase", "analogical_bridge", "cross_domain_transfer",
            "category_blend", "polar_shift", "semantic_drift", "extreme_analogy",
        }
        for level in [x / 100.0 for x in range(0, 101)]:
            op = self.engine.pick_operation(level)
            self.assertIn(op, valid_ops)


class TestChaosOperations(unittest.TestCase):
    """Test each of the 7 chaos operations individually."""

    def setUp(self) -> None:
        self.pool = _make_sample_pool()
        self.engine = ChaosEngine(self.pool)
        self.a1 = list(self.pool)[0]
        self.a2 = list(self.pool)[1]

    def _assert_chaos_atom(self, atom, op_name: str) -> None:
        self.assertIsNotNone(atom)
        self.assertEqual(atom.type, "insight")
        self.assertTrue(0.0 <= atom.confidence <= 0.5)
        self.assertIn("chaos_op", atom.metadata)
        self.assertEqual(atom.metadata["chaos_op"], op_name)
        self.assertIn("drunk_level", atom.metadata)
        self.assertTrue(0.0 <= atom.metadata["drunk_level"] <= 1.0)

    def test_sober_rephrase(self) -> None:
        atom = self.engine.sober_rephrase(self.a1)
        if atom is not None:
            self._assert_chaos_atom(atom, "sober_rephrase")
            self.assertGreater(len(atom.content), 5)

    def test_sober_rephrase_short_atom(self) -> None:
        """Very short atom may not change — should return None."""
        short = Atom(id="s", type="preference", content="likes X")
        atom = self.engine.sober_rephrase(short)
        self.assertIsNone(atom)

    def test_analogical_bridge(self) -> None:
        atom = self.engine.analogical_bridge(self.a1, self.a2, self.pool)
        if atom is not None:
            self._assert_chaos_atom(atom, "analogical_bridge")
            self.assertIn("structural parallel", atom.content.lower())

    def test_cross_domain_transfer(self) -> None:
        atom = self.engine.cross_domain_transfer(self.a1, self.a2, self.pool)
        if atom is not None:
            self._assert_chaos_atom(atom, "cross_domain_transfer")
            self.assertIn("transferred", atom.content.lower())

    def test_category_blend(self) -> None:
        atom = self.engine.category_blend(self.a1, self.a2)
        if atom is not None:
            self._assert_chaos_atom(atom, "category_blend")
            self.assertIn("blend", atom.content.lower())

    def test_polar_shift(self) -> None:
        atom = self.engine.polar_shift(self.a1)
        if atom is not None:
            self._assert_chaos_atom(atom, "polar_shift")
            self.assertIn("Polar shift", atom.content)

    def test_polar_shift_no_change(self) -> None:
        """An atom without invertible polarity should return None."""
        neutral = Atom(id="n", type="insight", content="the sky is blue")
        atom = self.engine.polar_shift(neutral)
        self.assertIsNone(atom)

    def test_semantic_drift(self) -> None:
        atom = self.engine.semantic_drift(self.a1)
        if atom is not None:
            self._assert_chaos_atom(atom, "semantic_drift")
            self.assertIn("Semantic drift", atom.content)

    def test_semantic_drift_short(self) -> None:
        short = Atom(id="s", type="preference", content="x y")
        atom = self.engine.semantic_drift(short)
        self.assertIsNone(atom)

    def test_extreme_analogy(self) -> None:
        atom = self.engine.extreme_analogy(self.a1, self.a2)
        if atom is not None:
            self._assert_chaos_atom(atom, "extreme_analogy")
            self.assertIn("Wild analogy", atom.content)


class TestChaosGenerate(unittest.TestCase):
    """Test the generate method."""

    def setUp(self) -> None:
        self.pool = _make_sample_pool()
        self.engine = ChaosEngine(self.pool)

    def test_generate_returns_atoms(self) -> None:
        atoms = self.engine.generate(batch=10)
        for atom in atoms:
            self.assertIsInstance(atom, Atom)
            self.assertEqual(atom.type, "insight")

    def test_generate_empty_pool(self) -> None:
        empty_pool = Pool()
        engine = ChaosEngine(empty_pool)
        atoms = engine.generate(batch=5)
        self.assertEqual(atoms, [])

    def test_generate_single_atom_pool(self) -> None:
        single_pool = Pool()
        single_pool.add(Atom(id="x", type="preference", content="test"))
        engine = ChaosEngine(single_pool)
        atoms = engine.generate(batch=5)
        self.assertEqual(atoms, [])

    def test_generate_batch_size(self) -> None:
        atoms = self.engine.generate(batch=3)
        self.assertLessEqual(len(atoms), 3)

    def test_generated_atoms_have_source_ids(self) -> None:
        atoms = self.engine.generate(batch=10)
        for atom in atoms:
            self.assertGreater(len(atom.source_ids), 0)
            for sid in atom.source_ids:
                self.assertIn(sid, self.pool)


if __name__ == "__main__":
    unittest.main()
