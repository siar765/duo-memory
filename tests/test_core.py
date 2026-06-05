"""
test_core.py — Tests for Atom, Pool, and generate_id.
"""

import json
import os
import tempfile
import unittest

from duo_memory.core import Atom, Pool, generate_id, ATOM_TYPES


class TestGenerateId(unittest.TestCase):
    """Tests for the generate_id helper."""

    def test_length(self) -> None:
        for _ in range(100):
            self.assertEqual(len(generate_id()), 8)

    def test_hex_chars(self) -> None:
        for _ in range(100):
            gid = generate_id()
            self.assertTrue(all(c in "0123456789abcdef" for c in gid))

    def test_uniqueness(self) -> None:
        ids = {generate_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)


class TestAtom(unittest.TestCase):
    """Tests for the Atom dataclass."""

    def test_create_default(self) -> None:
        atom = Atom(id="aabbccdd", type="preference", content="likes cats")
        self.assertEqual(atom.id, "aabbccdd")
        self.assertEqual(atom.type, "preference")
        self.assertEqual(atom.content, "likes cats")
        self.assertEqual(atom.confidence, 1.0)
        self.assertEqual(atom.evidence, "")
        self.assertEqual(atom.source_ids, [])
        self.assertIsInstance(atom.created_at, float)
        self.assertEqual(atom.metadata, {})

    def test_all_types_valid(self) -> None:
        for t in ATOM_TYPES:
            atom = Atom(id="x", type=t, content="test")
            self.assertEqual(atom.type, t)

    def test_invalid_type(self) -> None:
        with self.assertRaises(ValueError):
            Atom(id="x", type="invalid_type", content="test")

    def test_confidence_range(self) -> None:
        with self.assertRaises(ValueError):
            Atom(id="x", type="insight", content="test", confidence=1.5)
        with self.assertRaises(ValueError):
            Atom(id="x", type="insight", content="test", confidence=-0.1)

    def test_empty_id(self) -> None:
        with self.assertRaises(ValueError):
            Atom(id="", type="preference", content="test")

    def test_to_dict_roundtrip(self) -> None:
        atom = Atom(
            id="abc123", type="decision", content="chose X", confidence=0.8,
            evidence="log", source_ids=["src1"], metadata={"key": "val"},
        )
        d = atom.to_dict()
        self.assertEqual(d["id"], "abc123")
        self.assertEqual(d["type"], "decision")
        self.assertEqual(d["content"], "chose X")
        self.assertEqual(d["metadata"], {"key": "val"})

        atom2 = Atom.from_dict(d)
        self.assertEqual(atom, atom2)
        self.assertEqual(atom2.metadata, {"key": "val"})

    def test_equality(self) -> None:
        import time
        now = time.time()
        a1 = Atom(id="a", type="insight", content="hello", created_at=now)
        a2 = Atom(id="a", type="insight", content="hello", created_at=now)
        self.assertEqual(a1, a2)

    def test_inequality(self) -> None:
        a1 = Atom(id="a", type="insight", content="hello")
        a2 = Atom(id="b", type="insight", content="hello")
        self.assertNotEqual(a1, a2)


class TestPool(unittest.TestCase):
    """Tests for the Pool class."""

    def setUp(self) -> None:
        self.pool = Pool()
        self.a1 = Atom(id="a1", type="preference", content="likes python")
        self.a2 = Atom(id="a2", type="environment", content="lives in container")
        self.a3 = Atom(id="a3", type="decision", content="chose linux")

    def test_add_and_get(self) -> None:
        self.pool.add(self.a1)
        self.assertEqual(self.pool.get("a1"), self.a1)

    def test_add_overwrite(self) -> None:
        self.pool.add(self.a1)
        replacement = Atom(id="a1", type="preference", content="likes rust")
        self.pool.add(replacement)
        self.assertEqual(self.pool.get("a1").content, "likes rust")

    def test_add_many(self) -> None:
        self.pool.add_many([self.a1, self.a2, self.a3])
        self.assertEqual(len(self.pool), 3)

    def test_remove(self) -> None:
        self.pool.add(self.a1)
        self.pool.remove("a1")
        self.assertEqual(len(self.pool), 0)

    def test_remove_missing(self) -> None:
        with self.assertRaises(KeyError):
            self.pool.remove("nonexistent")

    def test_get_missing(self) -> None:
        with self.assertRaises(KeyError):
            self.pool.get("nonexistent")

    def test_len(self) -> None:
        self.assertEqual(len(self.pool), 0)
        self.pool.add(self.a1)
        self.assertEqual(len(self.pool), 1)
        self.pool.add(self.a2)
        self.assertEqual(len(self.pool), 2)

    def test_iter(self) -> None:
        self.pool.add_many([self.a1, self.a2, self.a3])
        ids = {a.id for a in self.pool}
        self.assertEqual(ids, {"a1", "a2", "a3"})

    def test_contains(self) -> None:
        self.pool.add(self.a1)
        self.assertIn("a1", self.pool)
        self.assertNotIn("a2", self.pool)

    def test_random_sample(self) -> None:
        self.pool.add_many([self.a1, self.a2, self.a3])
        sampled = self.pool.random(2)
        self.assertEqual(len(sampled), 2)
        for a in sampled:
            self.assertIn(a.id, {"a1", "a2", "a3"})

    def test_random_zero(self) -> None:
        self.pool.add(self.a1)
        self.assertEqual(self.pool.random(0), [])

    def test_random_more_than_pool(self) -> None:
        self.pool.add(self.a1)
        sampled = self.pool.random(10)
        self.assertEqual(len(sampled), 1)

    def test_by_type(self) -> None:
        self.pool.add_many([
            self.a1,  # preference
            Atom(id="a4", type="preference", content="likes vim"),
            self.a2,  # environment
        ])
        prefs = self.pool.by_type("preference")
        self.assertEqual(len(prefs), 2)
        envs = self.pool.by_type("environment")
        self.assertEqual(len(envs), 1)
        self.assertEqual(self.pool.by_type("insight"), [])

    def test_by_type_invalid(self) -> None:
        with self.assertRaises(ValueError):
            self.pool.by_type("nope")

    def test_stats_empty(self) -> None:
        s = self.pool.stats()
        self.assertEqual(s["count"], 0)
        self.assertEqual(s["types"], {})
        self.assertEqual(s["avg_confidence"], 0.0)

    def test_stats(self) -> None:
        self.pool.add_many([
            Atom(id="a1", type="preference", content="x", confidence=1.0),
            Atom(id="a2", type="environment", content="y", confidence=0.5),
        ])
        s = self.pool.stats()
        self.assertEqual(s["count"], 2)
        self.assertEqual(s["types"]["preference"], 1)
        self.assertEqual(s["types"]["environment"], 1)
        self.assertEqual(s["avg_confidence"], 0.75)

    def test_search_empty(self) -> None:
        self.assertEqual(self.pool.search("anything"), [])

    def test_search_found(self) -> None:
        self.pool.add_many([
            self.a1,  # "likes python"
            self.a2,  # "lives in container"
            self.a3,  # "chose linux"
        ])
        results = self.pool.search("python")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn(self.a1, results)

    def test_search_all_on_empty_query(self) -> None:
        self.pool.add_many([self.a1, self.a2])
        results = self.pool.search("")
        self.assertEqual(len(results), 2)

    def test_jsonl_roundtrip(self) -> None:
        self.pool.add_many([self.a1, self.a2, self.a3])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            fpath = f.name

        try:
            self.pool.to_jsonl(fpath)
            pool2 = Pool.from_jsonl(fpath)
            self.assertEqual(len(pool2), 3)
            self.assertEqual(pool2.get("a1").content, "likes python")
            self.assertEqual(pool2.get("a3").content, "chose linux")
        finally:
            os.unlink(fpath)

    def test_jsonl_empty(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            fpath = f.name

        try:
            self.pool.to_jsonl(fpath)
            pool2 = Pool.from_jsonl(fpath)
            self.assertEqual(len(pool2), 0)
        finally:
            os.unlink(fpath)


if __name__ == "__main__":
    unittest.main()
