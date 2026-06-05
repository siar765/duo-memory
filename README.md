<div align="center">

# 🧠 Duo-Memory

**A dual-pool memory engine that generates insights through controlled randomness**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-71%20passed-green)]()

</div>

---

## The Idea

Every memory system stores **clean facts**. But insights don't come from clean facts alone — they come from **unexpected connections** between them.

Duo-Memory simulates how human memory generates creative ideas: one pool holds structured, factual memories; another deliberately mixes them with semi-random operations — analogy, inversion, drift — like a mind wandering after a drink. A critic gate filters out nonsense while letting through surprising-but-logical connections.

The result: **1 + 1 ≠ 789, but 1 + 1 might be 🥂** (two drinks equal a good conversation).

```
Clean Pool ──→ Chaos Engine ──→ Critic Gate ──→ Insight Pool
(structure)      (7 ops)          (6-dim score)    (feedback loop)
```

---

## Quick Start

```python
# No pip install needed — just drop the source in your project
from duo_memory import Pool, ChaosEngine, Critic, generate_id

# 1. Create a clean pool with some facts
pool = Pool()
pool.add_many([
    {
        "id": generate_id(),
        "type": "preference",
        "content": "User prefers CLI over GUI",
        "confidence": 0.9,
        "evidence": "observed"
    },
    {
        "id": generate_id(),
        "type": "environment",
        "content": "Docker has port 443 blocked",
        "confidence": 0.8,
        "evidence": "tested"
    },
])

# 2. Run the chaos engine — it gets "drunk" at random levels
engine = ChaosEngine(pool)
chaos_atoms = engine.generate(batch=5)

# 3. Filter with the critic gate
critic = Critic(pool)
results = critic.batch_evaluate(chaos_atoms)
for r in results:
    if r["valid"]:
        print(f"✅ [{r['score']}/6] {r['atom'].content[:60]}")

# 4. Inject back
valid = critic.filter_valid(results)
pool.add_many(valid)
```

Output:
```
✅ [4/6] CLI is like minimalist coffee: pure and direct
✅ [5/6] What if Docker restrictions were a feature, not a bug?
✅ [4/6] The port 443 problem and the CLI preference share the same root cause
```

---

## Architecture

### Two Pools, One Loop

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Clean Pool    │────→│   Chaos Engine   │────→│   Critic Gate   │
│ (structured AT) │     │ (7 "drunk" ops)  │     │ (6-dim scoring) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        ↑                                                │
        │                   ┌──────────────────┐          │
        └───────────────────│   Insight Pool   │←─────────┘
                             │    (feedback)    │
                             └──────────────────┘
```

### The 7 Chaos Operations

Each operation corresponds to a different "level of intoxication":

| Operation | Drunk Level | What it does |
|-----------|-------------|-------------|
| `sober_rephrase` | 0.0 - 0.3 | Minimal synonym swap, almost no change |
| `analogical_bridge` | 0.1 - 0.7 | Finds structural similarity between domains |
| `cross_domain_transfer` | 0.2 - 0.8 | Applies pattern from domain A to domain B |
| `category_blend` | 0.3 - 0.9 | Mixes type and content across facts |
| `polar_shift` | 0.4 - 1.0 | Inverts a fact productively |
| `semantic_drift` | 0.5 - 1.0 | Shifts one attribute, keeps the logic |
| `extreme_analogy` | 0.7 - 1.0 | Far-fetched comparison that works |

Operation selection is driven by a Beta(0.5, 0.5) distribution — it naturally swings between sober and drunk states, like real intoxication patterns.

### The Critic Gate (6 Dimensions)

Every chaos output is scored on 6 dimensions. Must score ≥ 3 to pass:

1. **Not nonsense** — no 1+1=789, no absolute statements (instant reject)
2. **Not empty** — no 套话/formulaic talk (instant reject)
3. **Not duplicate** — must differ from existing atoms
4. **Sources valid** — all referenced atoms must exist
5. **Actually changed** — must differ from source content
6. **Insight signals** — contains keywords suggesting useful connection

Bonus: if drunk_level ≥ 0.6 and score ≥ 3, +1 point (drunk-but-logical is the most valuable).

---

## CLI Usage

```bash
# Generate chaos + run critic
duo-memory run --batch 5

# Inject passed chaos back into pool
duo-memory inject

# Pool stats
duo-memory stats

# Search atoms
duo-memory search "CLI"

# Run demo with sample data
duo-memory demo
```

---

## Why Two Pools?

Existing memory systems (MemGPT, Graphiti, Hy-Memory) all optimize for **accuracy** — retrieving the right fact at the right time. That's important, but it misses half of what memory does: **generating new ideas**.

Duo-Memory is the first system that intentionally introduces noise (through controlled, semi-random operations) and then filters it for value. It's not a replacement for structured memory — it's a complement that runs alongside it.

| | Structured Memory (agent-memory) | Duo-Memory |
|---|---|---|
| **Goal** | Recall facts accurately | Generate insights |
| **Operations** | Search, inject, forget | Analogize, invert, drift |
| **Noise** | Removed | Controlled |
| **Output** | Facts | Hypotheses |

---

## Why "Duo"?

The name reflects the core mechanism — two pools working together, like the brain's left and right hemispheres. One stores and retrieves accurately; the other wanders, makes unexpected connections, and feeds back what's worth keeping.

---

## The Backstory

This was born from a late-night conversation: "What if an AI had a 'drunk' memory pool — not random nonsense, but the kind of loose associative thinking that leads to real insights?"

The hypothesis was that **controlled chaos** in memory could produce the same kind of creative leaps humans make when they're not thinking too rigidly. The first prototype (v0.1) was a simple text splicer. v0.2 added the drunk-but-logical operations and the critic gate. This repo is the clean, packaged version — ready to run alongside any structured memory system.

---

## License

MIT
