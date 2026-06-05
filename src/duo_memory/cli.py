"""
cli.py — argparse-based command-line interface for duo-memory.

Provides commands for generating chaos, inspecting pool state,
searching, injecting feedback, and running a demo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from .core import Atom, Pool, generate_id
from .chaos import ChaosEngine
from .critic import Critic

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_HOME = os.path.expanduser("~/.duo-memory")
_DEFAULT_ATOMS_DIR = os.path.join(_DEFAULT_HOME, "atoms")
_DEFAULT_FEEDBACK_PATH = os.path.join(_DEFAULT_HOME, "feedback.jsonl")


def _ensure_default_dirs() -> None:
    """Create default storage directories if they don't exist."""
    os.makedirs(_DEFAULT_ATOMS_DIR, exist_ok=True)


def _load_main_pool() -> Pool:
    """Load the main pool from the default atoms directory.

    Merges all JSONL files found in the directory.
    """
    pool = Pool()
    if os.path.isdir(_DEFAULT_ATOMS_DIR):
        for fname in sorted(os.listdir(_DEFAULT_ATOMS_DIR)):
            if fname.endswith(".jsonl"):
                fpath = os.path.join(_DEFAULT_ATOMS_DIR, fname)
                try:
                    chunk = Pool.from_jsonl(fpath)
                    pool.add_many(list(chunk))
                except Exception:
                    pass  # skip corrupt files
    return pool


def _save_main_pool(pool: Pool) -> None:
    """Save the entire pool to the default atoms directory."""
    _ensure_default_dirs()
    main_path = os.path.join(_DEFAULT_ATOMS_DIR, "main.jsonl")
    pool.to_jsonl(main_path)


def _load_feedback() -> List[Dict]:
    """Load feedback entries from the feedback queue file."""
    if not os.path.exists(_DEFAULT_FEEDBACK_PATH):
        return []
    entries: List[Dict] = []
    with open(_DEFAULT_FEEDBACK_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _append_feedback(entry: Dict) -> None:
    """Append a single feedback entry to the queue file."""
    _ensure_default_dirs()
    with open(_DEFAULT_FEEDBACK_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Generate chaos atoms, run critic, show results, save valid ones."""
    if args.agent_pool:
        pool = Pool.load_from_agent_memory(args.agent_pool)
        print(f"Loaded {len(pool)} atom(s) from agent-memory pool (read-only).")
    else:
        pool = _load_main_pool()
    if len(pool) < 2:
        print("Need at least 2 atoms in pool to run chaos generation.", file=sys.stderr)
        print(f"Pool has {len(pool)} atom(s).", file=sys.stderr)
        return 1

    engine = ChaosEngine(pool)
    critic = Critic(pool, mode=args.mode)

    batch = max(1, args.batch)
    generated = engine.generate(batch=batch)

    if not generated:
        print("No chaos atoms generated (pool too small or all ops returned None).")
        return 0

    results = critic.batch_evaluate(generated)
    valid_atoms = [a for a, r in zip(generated, results) if r["valid"]]

    print(f"Generated: {len(generated)} chaos atoms")
    print(f"Valid:     {len(valid_atoms)} passed critic gate")
    print()

    for atom, result in zip(generated, results):
        status = "✓" if result["valid"] else "✗"
        print(f"  [{status}] score={result['score']:.1f}  {atom.content[:80]}...")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"         reason: {r}")

    # Save valid atoms to feedback queue
    for atom in valid_atoms:
        _append_feedback({
            "action": "propose",
            "atom": atom.to_dict(),
        })

    if valid_atoms:
        print(f"\nSaved {len(valid_atoms)} valid atom(s) to feedback queue at:")
        print(f"  {_DEFAULT_FEEDBACK_PATH}")

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show pool statistics."""
    pool = _load_main_pool()
    s = pool.stats()
    print(f"Total atoms: {s['count']}")
    print(f"Average confidence: {s['avg_confidence']}")
    print("\nType distribution:")
    for tname, count in sorted(s["types"].items()):
        bar = "█" * count
        print(f"  {tname:15s} {count:4d}  {bar}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search atoms by query string."""
    pool = _load_main_pool()
    results = pool.search(args.query)
    if not results:
        print(f"No results for query: {args.query}")
        return 0
    print(f"Found {len(results)} result(s) for: {args.query}\n")
    for atom in results:
        print(f"  [{atom.id}] {atom.type:15s} ({atom.confidence:.2f})")
        print(f"  {atom.content[:120]}")
        print()
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    """Inject feedback queue entries into the pool."""
    entries = _load_feedback()
    if not entries:
        print("Feedback queue is empty.")
        return 0

    pool = _load_main_pool()
    accepted = 0
    rejected = 0

    for entry in entries:
        action = entry.get("action", "")
        atom_dict = entry.get("atom")
        if action == "propose" and atom_dict:
            atom = Atom.from_dict(atom_dict)
            pool.add(atom)
            accepted += 1
        else:
            rejected += 1

    _save_main_pool(pool)

    # Clear feedback file
    _ensure_default_dirs()
    with open(_DEFAULT_FEEDBACK_PATH, "w", encoding="utf-8") as fh:
        fh.write("")

    print(f"Injected {accepted} atom(s) into pool.")
    if rejected:
        print(f"Skipped {rejected} unrecognized feedback entry(ies).")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a quick demo with sample atoms."""
    pool = Pool()

    sample_facts = [
        ("preference", "Prefers functional programming over OOP for data pipelines"),
        ("environment", "Works primarily in Python with occasional Rust for performance"),
        ("decision", "Chose SQLite over PostgreSQL for single-user analytics projects"),
        ("rejection", "Rejected microservices architecture in favor of monolith-first"),
        ("pattern", "Spends first 90 minutes of the day on deep work before meetings"),
        ("relationship", "Collaborates better when async communication is the default"),
        ("insight", "Code readability is a form of communication, not just aesthetics"),
        ("preference", "Uses terminal-based tools over GUI whenever possible"),
    ]

    for tname, content in sample_facts:
        pool.add(Atom(
            id=generate_id(),
            type=tname,
            content=content,
            confidence=0.9,
        ))

    print("=== Sample Pool ===")
    print(f"Loaded {len(pool)} atoms\n")
    stats = pool.stats()
    print(f"Types: {dict(stats['types'])}")
    print(f"Avg confidence: {stats['avg_confidence']}\n")

    # Chaos generation
    print("=== Chaos Generation ===")
    engine = ChaosEngine(pool)
    generated = engine.generate(batch=5)
    print(f"Generated {len(generated)} chaos atoms\n")

    # Critic evaluation
    print("=== Critic Evaluation ===")
    critic = Critic(pool)
    results = critic.batch_evaluate(generated)
    valid = critic.filter_valid(generated, results)

    for atom, result in zip(generated, results):
        status = "✓" if result["valid"] else "✗"
        op = atom.metadata.get("chaos_op", "?")
        drunk = atom.metadata.get("drunk_level", 0.0)
        print(f"  [{status}] [{op}] drunk={drunk:.2f} score={result['score']:.1f}")
        print(f"        {atom.content[:100]}")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"        → {r}")
        print()

    print(f"Valid: {len(valid)} / {len(generated)} passed critic gate")

    if valid:
        print("\nValid insight atoms:")
        for i, atom in enumerate(valid, 1):
            print(f"  {i}. [{atom.confidence:.2f}] {atom.content[:120]}")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duo-memory",
        description="Dual-pool memory system: clean facts + chaotic insights",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p_run = sub.add_parser("run", help="Generate chaos atoms → critic gate → feedback queue")
    p_run.add_argument(
        "--batch", type=int, default=5,
        help="Number of chaos atoms to generate (default: 5)",
    )
    p_run.add_argument(
        "--agent-pool", type=str, default=None,
        help="Path to agent-memory atoms directory (read-only; overrides default pool)",
    )
    p_run.add_argument(
        "--mode", type=str, choices=["strict", "normal", "lenient"], default="normal",
        help="Critic gate mode: strict (5.0), normal (4.0, default), lenient (3.0)",
    )

    # stats
    sub.add_parser("stats", help="Show pool statistics")

    # search
    p_search = sub.add_parser("search", help="Search atoms by keyword")
    p_search.add_argument("query", type=str, help="Search query string")

    # inject
    sub.add_parser("inject", help="Inject feedback queue into main pool")

    # demo
    sub.add_parser("demo", help="Run a quick demo with sample atoms")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Parses args and dispatches to command handlers."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    _ensure_default_dirs()

    handlers = {
        "run": cmd_run,
        "stats": cmd_stats,
        "search": cmd_search,
        "inject": cmd_inject,
        "demo": cmd_demo,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
