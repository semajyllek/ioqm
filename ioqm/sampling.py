"""
Choose images for human labelling.

Most images need no human: both counters agree. The informative ones are where
they disagree. But agreement must be sampled too, otherwise the rate at which
both counters are wrong in the same way (false agreement) cannot be measured,
and that rate bounds every counter-scored result.

Each image of a run, scored by two counters A and B, falls in one stratum:
  disagree         A and B give different prompt_exact verdicts
  both_exact       both say every count is right
  both_wrong_same  both say wrong, with identical counts for the requested classes
  both_wrong_diff  both say wrong, with different counts

Within each (run, stratum), images are drawn uniformly at random, at most one
seed per prompt until prompts run out. Each sampled image carries an
inverse-probability weight N_stratum / n_stratum, so weighted estimates
(ioqm.agreement) describe the whole run.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

STRATA = ("disagree", "both_exact", "both_wrong_same", "both_wrong_diff")
DEFAULT_ALLOC = {"disagree": 0.45, "both_exact": 0.25, "both_wrong_same": 0.10, "both_wrong_diff": 0.20}


def stratum(a: Mapping, b: Mapping) -> str:
    if a["prompt_exact"] != b["prompt_exact"]:
        return "disagree"
    if a["prompt_exact"]:
        return "both_exact"
    same = all(int(a["detected"].get(c, 0)) == int(b["detected"].get(c, 0)) for c in a["expected"])
    return "both_wrong_same" if same else "both_wrong_diff"


def _allocate(n: int, alloc: Mapping[str, float], available: Mapping[str, int]) -> Dict[str, int]:
    """Split n across strata by `alloc`, capped by what is available; leftover goes
    to strata that still have images, in the order of STRATA."""
    want = {s: int(math.floor(n * alloc.get(s, 0.0))) for s in STRATA}
    want[STRATA[0]] += n - sum(want.values())
    got = {s: min(want[s], available.get(s, 0)) for s in STRATA}
    left = n - sum(got.values())
    while left > 0:
        moved = False
        for s in STRATA:
            if left and got[s] < available.get(s, 0):
                got[s] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return got


def _spread(rows: List[Mapping], k: int, rng: random.Random) -> List[Mapping]:
    """k rows drawn at random, preferring distinct prompts."""
    rows = rows[:]
    rng.shuffle(rows)
    seen, first, rest = set(), [], []
    for r in rows:
        (rest if r["prompt_id"] in seen else first).append(r)
        seen.add(r["prompt_id"])
    return (first + rest)[:k]


def item_id(run_dir: str, image: str) -> str:
    return hashlib.sha1(f"{run_dir}|{image}".encode()).hexdigest()[:12]


def sample_run(
    run_dir: str,
    a_rows: Sequence[Mapping],
    b_rows: Sequence[Mapping],
    n: int,
    a_name: str,
    b_name: str,
    alloc: Mapping[str, float] = DEFAULT_ALLOC,
    seed: int = 0,
    exclude: Sequence[str] = (),
) -> List[dict]:
    b_by = {r["image"]: r for r in b_rows}
    excl = set(exclude)
    pools: Dict[str, List[Tuple[Mapping, Mapping]]] = {s: [] for s in STRATA}
    for a in a_rows:
        b = b_by.get(a["image"])
        if b is None or item_id(run_dir, a["image"]) in excl:
            continue
        pools[stratum(a, b)].append((a, b))
    take = _allocate(n, alloc, {s: len(v) for s, v in pools.items()})
    rng = random.Random(f"{seed}|{run_dir}")
    items = []
    for s in STRATA:
        if not take[s]:
            continue
        by_img = {a["image"]: (a, b) for a, b in pools[s]}
        picked = _spread([a for a, _ in pools[s]], take[s], rng)
        weight = len(pools[s]) / take[s]
        for a in picked:
            _, b = by_img[a["image"]]
            items.append({
                "item_id": item_id(run_dir, a["image"]),
                "run_dir": run_dir,
                "image": a["image"],
                "prompt_id": a["prompt_id"],
                "model": a.get("model"),
                "split": a.get("split"),
                "n_types": a.get("n_types"),
                "max_count": a.get("max_count"),
                "classes": list(a["expected"]),        # shown to the labeller
                "expected": a["expected"],             # NOT shown to the labeller
                "counters": {a_name: {c: int(a["detected"].get(c, 0)) for c in a["expected"]},
                             b_name: {c: int(b["detected"].get(c, 0)) for c in a["expected"]}},
                "stratum": s,
                "stratum_size": len(pools[s]),
                "weight": weight,
            })
    return items


def interleave(items_by_run: Mapping[str, List[dict]], seed: int = 0) -> List[dict]:
    """One queue in random order, so a partly finished labelling session still covers
    every run and stratum."""
    all_items = [it for v in items_by_run.values() for it in v]
    random.Random(seed).shuffle(all_items)
    return all_items


def weights_for_run(items: Sequence[Mapping], run_dir: str) -> Dict[str, float]:
    return {it["image"]: it["weight"] for it in items if it["run_dir"] == run_dir}


def labeled_weights(items: Sequence[Mapping], run_dir: str, labeled_images: Sequence[str]) -> Dict[str, float]:
    """Inverse-probability weights recomputed over the images that actually got a usable
    label (skipped or ambiguous items shrink their stratum's sample)."""
    lab = set(labeled_images)
    n: Dict[str, int] = {}
    for it in items:
        if it["run_dir"] == run_dir and it["image"] in lab:
            n[it["stratum"]] = n.get(it["stratum"], 0) + 1
    return {it["image"]: it["stratum_size"] / n[it["stratum"]]
            for it in items if it["run_dir"] == run_dir and it["image"] in lab}
