"""
demo.py — Quick demonstration of the duo-memory system.

Creates a pool with 8 sample facts about a fictional user,
runs chaos generation, applies the critic gate, and prints results.
"""

import sys
import os

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from duo_memory.core import Atom, Pool, generate_id
from duo_memory.chaos import ChaosEngine
from duo_memory.critic import Critic


def main() -> None:
    """Run the duo-memory demo pipeline."""
    pool = Pool()

    # 8 sample facts about a fictional user
    sample_facts = [
        Atom(
            id=generate_id(),
            type="preference",
            content="Prefers functional programming over OOP for data pipelines",
            confidence=0.90,
            evidence="From user interview notes",
        ),
        Atom(
            id=generate_id(),
            type="environment",
            content="Works primarily in Python with occasional Rust for performance-critical sections",
            confidence=0.95,
            evidence="Project repository analysis",
        ),
        Atom(
            id=generate_id(),
            type="decision",
            content="Chose SQLite over PostgreSQL for single-user analytics projects",
            confidence=0.85,
            evidence="Tech stack documentation",
        ),
        Atom(
            id=generate_id(),
            type="rejection",
            content="Rejected microservices architecture in favor of monolith-first approach",
            confidence=0.80,
            evidence="Architecture decision record",
        ),
        Atom(
            id=generate_id(),
            type="pattern",
            content="Spends the first 90 minutes of each day on deep work before checking messages",
            confidence=0.75,
            evidence="Self-reported time tracking",
        ),
        Atom(
            id=generate_id(),
            type="relationship",
            content="Collaborates better when async communication (docs, PRs) is the default",
            confidence=0.88,
            evidence="Team retrospective feedback",
        ),
        Atom(
            id=generate_id(),
            type="insight",
            content="Code readability is a form of communication, not just aesthetics",
            confidence=0.92,
            evidence="Personal blog post",
        ),
        Atom(
            id=generate_id(),
            type="preference",
            content="Uses terminal-based tools (vim, tmux, ripgrep) over GUI whenever possible",
            confidence=0.85,
            evidence="Desktop setup screenshot",
        ),
    ]

    for atom in sample_facts:
        pool.add(atom)

    print("=" * 60)
    print("  Duo-Memory Demo")
    print("=" * 60)
    print(f"\nPool loaded: {len(pool)} atoms")

    # Show type distribution
    stats = pool.stats()
    print(f"\nType distribution:")
    for tname, count in sorted(stats["types"].items()):
        print(f"  {tname:15s}: {count}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")

    # --- Chaos Generation ---
    print("\n" + "=" * 60)
    print("  Chaos Generation (batch=5)")
    print("=" * 60)

    engine = ChaosEngine(pool)
    generated = engine.generate(batch=5)
    print(f"\nGenerated {len(generated)} chaos atoms:\n")

    for i, atom in enumerate(generated, 1):
        op = atom.metadata.get("chaos_op", "?")
        drunk = atom.metadata.get("drunk_level", 0.0)
        print(f"  [{i}] {op} (drunk={drunk:.2f}, conf={atom.confidence:.2f})")
        print(f"      {atom.content}")
        print()

    # --- Critic Evaluation ---
    print("=" * 60)
    print("  Critic Evaluation")
    print("=" * 60)

    critic = Critic(pool)
    results = critic.batch_evaluate(generated)
    valid_atoms = critic.filter_valid(generated, results)

    print(f"\nResults: {len(valid_atoms)} / {len(generated)} passed\n")

    for atom, result in zip(generated, results):
        status = "✓ PASS" if result["valid"] else "✗ FAIL"
        print(f"  {status}  score={result['score']:.1f}/6.0")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"       → {r}")
        print(f"       {atom.content[:100]}")
        print()

    if valid_atoms:
        print("=" * 60)
        print("  Valid Insight Atoms (candidate for injection)")
        print("=" * 60)
        for i, atom in enumerate(valid_atoms, 1):
            print(f"\n  {i}. [{atom.confidence:.2f}] {atom.content}")

    print("\nDone.")


if __name__ == "__main__":
    main()
