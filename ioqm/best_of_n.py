"""
Best-of-N selection with a verifier counter, scored by a DIFFERENT evaluator counter.

Setup: each prompt is generated with seeds 0..N-1 (one run directory), and every
image is detected by several counters (ioqm.detect stores the raw detections).
For each k in {1, 2, 4, ...}, the N seeds are split into N//k disjoint
consecutive groups. In each group the verifier picks one image, and the
evaluator scores that pick. So one N=16 generation run gives the whole
accuracy-vs-k curve, and smaller k gets more groups (lower variance).

Selection key (larger is better), applied to the VERIFIER's scores:
    (prompt_exact, type_soft, -extra_class, -seed)
The final -seed breaks ties toward the lowest seed. Seeds are random, so this
is equivalent to a random pick among the tied images.

Reported for each k, all on the EVALUATOR's scores unless noted:
    selected        count accuracy of the verifier's pick. THE RESULT.
    verifier_self   the same pick scored by the verifier itself. Optimistic by
                    construction; the gap to `selected` shows how much the
                    verifier is being gamed.
    oracle          the pick the evaluator itself would make (= pass@k under the
                    evaluator). The ceiling for a perfect verifier.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .metrics import Estimate, bootstrap, paired_bootstrap_diff


def selection_key(row: Mapping) -> Tuple:
    return (row["prompt_exact"], row["type_soft"], -row["extra_class"], -row["seed"])


def _index(rows: Iterable[Mapping]) -> Dict[str, Dict[int, Mapping]]:
    idx: Dict[str, Dict[int, Mapping]] = {}
    for r in rows:
        idx.setdefault(r["prompt_id"], {})[int(r["seed"])] = r
    return idx


def select(
    verifier_rows: Sequence[Mapping],
    eval_rows: Sequence[Mapping],
    k: int,
) -> List[dict]:
    """One output row per (prompt, group): the evaluator's scores for the
    verifier's pick, plus verifier_self_* and oracle_* fields."""
    v, e = _index(verifier_rows), _index(eval_rows)
    out = []
    for pid in sorted(set(v) & set(e)):
        seeds = sorted(set(v[pid]) & set(e[pid]))
        for g in range(len(seeds) // k):
            group = seeds[g * k:(g + 1) * k]
            pick = max(group, key=lambda s: selection_key(v[pid][s]))
            oracle = max(group, key=lambda s: selection_key(e[pid][s]))
            er = e[pid][pick]
            out.append({
                "prompt_id": pid, "group": g, "k": k, "seed": pick,
                "split": er.get("split"), "n_types": er.get("n_types"), "max_count": er.get("max_count"),
                **{m: er[m] for m in ("prompt_exact", "type_exact", "type_soft", "abs_err", "signed_err", "extra_class")},
                "verifier_self_prompt_exact": v[pid][pick]["prompt_exact"],
                "oracle_prompt_exact": e[pid][oracle]["prompt_exact"],
            })
    return out


def curve(
    verifier_rows: Sequence[Mapping],
    eval_rows: Sequence[Mapping],
    ks: Sequence[int] = (1, 2, 4, 8, 16),
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[int, Dict[str, object]]:
    """Accuracy-vs-k. Each k > 1 also gets a paired bootstrap difference against k=1."""
    by_k = {k: select(verifier_rows, eval_rows, k) for k in ks}
    by_k = {k: rows for k, rows in by_k.items() if rows}
    if 1 not in by_k:
        raise ValueError("need k=1 rows (no images detected by both counters?)")
    res: Dict[int, Dict[str, object]] = {}
    for k, rows in by_k.items():
        r: Dict[str, object] = {
            "n_groups_per_prompt": len(rows) / len({x["prompt_id"] for x in rows}),
            "selected": bootstrap(rows, "prompt_exact", n_boot=n_boot, seed=seed),
            "verifier_self": bootstrap(rows, "verifier_self_prompt_exact", n_boot=n_boot, seed=seed),
            "oracle": bootstrap(rows, "oracle_prompt_exact", n_boot=n_boot, seed=seed),
            "abs_err": bootstrap(rows, "abs_err", n_boot=n_boot, seed=seed),
            "signed_err": bootstrap(rows, "signed_err", n_boot=n_boot, seed=seed),
        }
        if k != 1:
            r["gain_vs_k1"], r["gain_p"] = paired_bootstrap_diff(rows, by_k[1], "prompt_exact", n_boot=n_boot, seed=seed)
        res[k] = r
    return res


def format_curve(res: Mapping[int, Mapping[str, object]]) -> str:
    def f(e: Estimate) -> str:
        return f"{e.mean:.3f} [{e.lo:.3f},{e.hi:.3f}]"
    lines = [f"{'k':>3} | {'selected (eval)':>22} | {'gain vs k=1':>22} | {'verifier self':>22} | {'oracle/pass@k':>22} | {'MAE':>6} | {'bias':>6}"]
    for k in sorted(res):
        r = res[k]
        gain = f(r["gain_vs_k1"]) if "gain_vs_k1" in r else "-"
        lines.append(f"{k:>3} | {f(r['selected']):>22} | {gain:>22} | {f(r['verifier_self']):>22} | "
                     f"{f(r['oracle']):>22} | {r['abs_err'].mean:>6.3f} | {r['signed_err'].mean:>+6.3f}")
    return "\n".join(lines)
