"""
Over-optimization curves for best-of-k selection against a counter.

Question: when a PROXY counter picks the best of k images, does accuracy as
judged by an independent GOLD judge keep rising with k, or does the proxy's
own opinion pull away from the truth? (Best-of-k analogue of Gao et al. 2022,
"Scaling Laws for Reward Model Overoptimization".)

Estimator. Each prompt has n images (seeds). Rather than splitting seeds into
disjoint groups (one group per prompt at k = n, so a noisy curve), we use the
standard unbiased best-of-k estimator: sort the prompt's n images by the
proxy's selection key (a strict order, ties broken by seed). The image at
sorted position j (0-based, ascending) is the maximum of a uniformly random
k-subset with probability C(j, k-1) / C(n, k). So

    E[gold score of the proxy's pick among k] = sum_j gold_j * C(j, k-1) / C(n, k)

exactly, for every k <= n, using all n images. Each prompt yields one value
per k; the cluster bootstrap over prompts gives CIs. A partially finished run
is fine: prompts use whatever seeds both judges have scored.

Optimization pressure is reported as k and as the analytic KL divergence of
best-of-k sampling from the base distribution, log k - (k-1)/k.

Proxies may be single counters or an agreement ensemble ("ens:a+b"), whose
key is (min prompt_exact over members, mean type_soft, -mean extra_class, -seed).
The gold judge must not be one of the proxies (or an ensemble member):
otherwise the curve measures a counter grading itself.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .best_of_n import selection_key
from .metrics import bootstrap

ENSEMBLE_PREFIX = "ens:"
GOLD_METRICS = ("prompt_exact", "type_exact", "abs_err")


def bon_kl(k: int) -> float:
    return math.log(k) - (k - 1) / k


def members(proxy: str) -> List[str]:
    return proxy[len(ENSEMBLE_PREFIX):].split("+") if proxy.startswith(ENSEMBLE_PREFIX) else [proxy]


def check_independent(proxies: Sequence[str], gold: str) -> None:
    for p in proxies:
        if gold in members(p):
            raise ValueError(f"gold judge {gold!r} is (part of) proxy {p!r}: the curve would be circular")


def ensemble_rows(rows_by_member: Sequence[Sequence[Mapping]]) -> List[dict]:
    """Combine several counters' score rows (same images) into one proxy row per image."""
    idx = [{r["image"]: r for r in rows} for rows in rows_by_member]
    common = set(idx[0]).intersection(*idx[1:])
    out = []
    for im in sorted(common):
        rs = [d[im] for d in idx]
        base = rs[0]
        out.append({
            "image": im, "prompt_id": base["prompt_id"], "seed": base["seed"],
            "prompt_exact": min(float(r["prompt_exact"]) for r in rs),
            "type_soft": sum(float(r["type_soft"]) for r in rs) / len(rs),
            "extra_class": sum(float(r["extra_class"]) for r in rs) / len(rs),
        })
    return out


def bok_weights(n: int, k: int) -> List[float]:
    """P(sorted position j is the max of a random k-subset of n), j = 0..n-1 (ascending)."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    total = math.comb(n, k)
    return [math.comb(j, k - 1) / total for j in range(n)]


def _by_prompt(rows: Iterable[Mapping]) -> Dict[str, Dict[int, Mapping]]:
    out: Dict[str, Dict[int, Mapping]] = {}
    for r in rows:
        out.setdefault(r["prompt_id"], {})[int(r["seed"])] = r
    return out


def expected_best_of_k(
    proxy_rows: Sequence[Mapping],
    gold_rows: Sequence[Mapping],
    ks: Sequence[int],
    metrics: Sequence[str] = GOLD_METRICS,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    One row per (prompt, k) with:
      gold_<m>    expected gold score of the proxy's best-of-k pick   (THE RESULT)
      proxy_self  expected proxy prompt_exact of its own pick         (what the proxy believes)
      oracle      expected gold prompt_exact if gold picked itself    (ceiling, = pass@k-like)
    Also returns counts: prompts used, images dropped for lacking a gold score,
    and (prompt, k) pairs skipped because the prompt has fewer than k images.
    """
    P, G = _by_prompt(proxy_rows), _by_prompt(gold_rows)
    out: List[dict] = []
    info = {"prompts": 0, "images_without_gold": 0, "skipped_prompt_k": 0}
    for pid in sorted(P):
        seeds = sorted(set(P[pid]) & set(G.get(pid, {})))
        info["images_without_gold"] += len(P[pid]) - len(seeds)
        n = len(seeds)
        if n == 0:
            continue
        info["prompts"] += 1
        by_proxy = sorted(seeds, key=lambda s: selection_key(P[pid][s]))
        by_gold = sorted(seeds, key=lambda s: selection_key(G[pid][s]))
        g0 = G[pid][seeds[0]]
        for k in ks:
            if k > n:
                info["skipped_prompt_k"] += 1
                continue
            w = bok_weights(n, k)
            row = {"prompt_id": pid, "k": k, "n_seeds": n, "split": g0.get("split"),
                   "n_types": g0.get("n_types"), "max_count": g0.get("max_count")}
            for m in metrics:
                row[f"gold_{m}"] = sum(wj * float(G[pid][s][m]) for wj, s in zip(w, by_proxy))
            row["proxy_self"] = sum(wj * float(P[pid][s]["prompt_exact"]) for wj, s in zip(w, by_proxy))
            row["oracle"] = sum(wj * float(G[pid][s]["prompt_exact"]) for wj, s in zip(w, by_gold))
            out.append(row)
    return out, info


def curve(proxy_rows, gold_rows, ks, n_boot: int = 2000, seed: int = 0) -> Tuple[Dict[int, dict], Dict[str, int]]:
    rows, info = expected_best_of_k(proxy_rows, gold_rows, ks)
    res: Dict[int, dict] = {}
    for k in ks:
        rk = [r for r in rows if r["k"] == k]
        if not rk:
            continue
        res[k] = {"kl": bon_kl(k),
                  "gold": bootstrap(rk, "gold_prompt_exact", n_boot=n_boot, seed=seed),
                  "proxy_self": bootstrap(rk, "proxy_self", n_boot=n_boot, seed=seed),
                  "oracle": bootstrap(rk, "oracle", n_boot=n_boot, seed=seed),
                  "gold_abs_err": bootstrap(rk, "gold_abs_err", n_boot=n_boot, seed=seed)}
        res[k]["gap"] = res[k]["proxy_self"].mean - res[k]["gold"].mean
    return res, info


def format_curve(res: Mapping[int, Mapping]) -> str:
    def f(e):
        return f"{e.mean:.3f} [{e.lo:.3f},{e.hi:.3f}]"
    lines = [f"{'k':>4} {'KL':>6} | {'gold (result)':>21} | {'proxy self':>21} | {'gap':>6} | "
             f"{'gold oracle':>21} | {'gold MAE':>8} | prompts"]
    for k in sorted(res):
        r = res[k]
        lines.append(f"{k:>4} {r['kl']:>6.2f} | {f(r['gold']):>21} | {f(r['proxy_self']):>21} | "
                     f"{r['gap']:>+6.3f} | {f(r['oracle']):>21} | {r['gold_abs_err'].mean:>8.3f} | "
                     f"{r['gold'].n_prompts}")
    return "\n".join(lines)


def concrete_pick(proxy_by_seed: Mapping[int, Mapping], k: int) -> int:
    """The proxy's actual pick among seeds 0..k-1 (for human verification of real picks)."""
    cand = [s for s in sorted(proxy_by_seed) if s < k]
    if len(cand) < k:
        raise ValueError(f"need seeds 0..{k - 1}, have {len(cand)} of them")
    return max(cand, key=lambda s: selection_key(proxy_by_seed[s]))


def picks_queue(
    run_dir: str,
    proxy_rows: Mapping[str, Sequence[Mapping]],
    gold_rows: Sequence[Mapping],
    ks: Sequence[int] = (1, 64),
    n_prompts: int = 50,
    seed: int = 0,
) -> List[dict]:
    """
    Human check of real picks. For the same random prompts, take each proxy's
    actual pick among seeds 0..k-1 for each k. One queue item per distinct image;
    `roles` lists every (proxy, k) it stands for (at k=1 all proxies pick seed 0).
    The gold judge's and proxies' verdicts are stored for the report, never shown.
    """
    import random as _random

    from .sampling import item_id

    P = {name: _by_prompt(rows) for name, rows in proxy_rows.items()}
    G = _by_prompt(gold_rows)
    kmax = max(ks)
    eligible = sorted(pid for pid in G
                      if all(pid in P[n] and all(s in P[n][pid] for s in range(kmax)) for n in P)
                      and all(s in G[pid] for s in range(kmax)))
    rng = _random.Random(seed)
    chosen = rng.sample(eligible, min(n_prompts, len(eligible)))
    items: Dict[str, dict] = {}
    for pid in chosen:
        for name in P:
            for k in ks:
                s = concrete_pick(P[name][pid], k)
                g = G[pid][s]
                img = g["image"]
                it = items.setdefault(img, {
                    "item_id": item_id(run_dir, img), "run_dir": run_dir, "image": img,
                    "prompt_id": pid, "seed": s, "classes": list(g["expected"]), "expected": g["expected"],
                    "split": g.get("split"), "n_types": g.get("n_types"), "max_count": g.get("max_count"),
                    "stratum": "picks", "gold_prompt_exact": float(g["prompt_exact"]), "roles": [],
                })
                it["roles"].append({"proxy": name, "k": k,
                                    "proxy_prompt_exact": float(P[name][pid][s]["prompt_exact"])})
    out = list(items.values())
    rng.shuffle(out)
    return out
