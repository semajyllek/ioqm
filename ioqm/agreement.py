"""
Agreement between two scorers (counter vs counter, or counter vs human).

Inputs are score rows from ioqm.detect.score_run, joined on image. The
reference (usually "human") is treated as ground truth.

Per pair of rows, on the requested classes only:
  verdict_agree   both give the same prompt_exact verdict.
  count_agree     fraction of requested types where the two counts are equal.
  abs_diff        mean |counter - reference| over requested types.
  signed_diff     mean (counter - reference). Negative means the counter undercounts.

Aggregates:
  accuracy        P(verdict agrees)
  kappa           Cohen's kappa on the prompt_exact verdict
  precision       P(reference says exact | counter says exact). This is what
                  matters for a best-of-N verifier or an RL reward: how often a
                  "correct" verdict is really correct.
  recall          P(counter says exact | reference says exact)
  counter_rate / reference_rate
                  the counts-accuracy each side reports; their difference is the
                  bias a counter introduces into a headline number.

Optional `weights` (image -> weight) give inverse-probability weights when the
images were drawn by a stratified sampler (ioqm.sampling), so the estimates
describe the full run, not the oversampled disagreements.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def join(a_rows: Iterable[Mapping], ref_rows: Iterable[Mapping]) -> List[Tuple[Mapping, Mapping]]:
    ref = {r["image"]: r for r in ref_rows}
    return [(a, ref[a["image"]]) for a in a_rows if a["image"] in ref]


def pair_stats(a: Mapping, ref: Mapping) -> dict:
    exp = ref["expected"]
    da, dr = a["detected"], ref["detected"]
    diffs = [int(da.get(c, 0)) - int(dr.get(c, 0)) for c in exp]
    n = len(exp)
    return {
        "image": a["image"],
        "a_exact": float(a["prompt_exact"]),
        "ref_exact": float(ref["prompt_exact"]),
        "verdict_agree": float(a["prompt_exact"] == ref["prompt_exact"]),
        "count_agree": sum(d == 0 for d in diffs) / n,
        "abs_diff": sum(abs(d) for d in diffs) / n,
        "signed_diff": sum(diffs) / n,
    }


def _wmean(xs: Sequence[float], ws: Sequence[float]) -> float:
    tw = sum(ws)
    return sum(x * w for x, w in zip(xs, ws)) / tw if tw else math.nan


def _summary(stats: Sequence[Mapping], ws: Sequence[float]) -> Dict[str, float]:
    a = [s["a_exact"] for s in stats]
    r = [s["ref_exact"] for s in stats]
    p11 = _wmean([x * y for x, y in zip(a, r)], ws)
    pa, pr = _wmean(a, ws), _wmean(r, ws)
    po = _wmean([s["verdict_agree"] for s in stats], ws)
    pe = pa * pr + (1 - pa) * (1 - pr)
    return {
        "accuracy": po,
        "kappa": (po - pe) / (1 - pe) if pe < 1 else math.nan,
        "precision": p11 / pa if pa else math.nan,
        "recall": p11 / pr if pr else math.nan,
        "counter_rate": pa,
        "reference_rate": pr,
        "count_agree": _wmean([s["count_agree"] for s in stats], ws),
        "abs_diff": _wmean([s["abs_diff"] for s in stats], ws),
        "signed_diff": _wmean([s["signed_diff"] for s in stats], ws),
    }


STAT_NAMES = ("accuracy", "kappa", "precision", "recall", "counter_rate", "reference_rate",
              "count_agree", "abs_diff", "signed_diff")


def agreement(
    a_rows: Iterable[Mapping],
    ref_rows: Iterable[Mapping],
    weights: Optional[Mapping[str, float]] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Tuple[float, float, float]]:
    """{stat: (point, lo, hi)} with a 95% bootstrap CI over images, plus "n"."""
    pairs = join(a_rows, ref_rows)
    if not pairs:
        raise ValueError("no images in common")
    stats = [pair_stats(a, r) for a, r in pairs]
    ws = [float(weights.get(s["image"], 0.0)) if weights else 1.0 for s in stats]
    point = _summary(stats, ws)
    rng = random.Random(seed)
    boots: Dict[str, List[float]] = {k: [] for k in point}
    idx = range(len(stats))
    for _ in range(n_boot):
        pick = [rng.choice(idx) for _ in idx]
        b = _summary([stats[i] for i in pick], [ws[i] for i in pick])
        for k, v in b.items():
            if not math.isnan(v):
                boots[k].append(v)

    def q(xs: List[float], p: float) -> float:
        if not xs:
            return math.nan
        xs = sorted(xs)
        return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]

    out = {k: (v, q(boots[k], 0.025), q(boots[k], 0.975)) for k, v in point.items()}
    out["n"] = (float(len(stats)), math.nan, math.nan)
    return out


def format_agreement(name: str, res: Mapping[str, Tuple[float, float, float]]) -> str:
    parts = [f"{name:28s} n={int(res['n'][0]):5d}"]
    for k in STAT_NAMES:
        v, lo, hi = res[k]
        parts.append(f"{k}={v:.3f}[{lo:.3f},{hi:.3f}]")
    return "  ".join(parts)


def threshold_sweep(
    dets: Mapping[str, Sequence[Mapping]],
    ref_rows: Iterable[Mapping],
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> List[dict]:
    """For a counter's raw detections, agreement with the reference at each global
    threshold. Pick the threshold by kappa or count_agree; use cross-validation
    before reporting a tuned number."""
    from .detect import count_detections
    from .metrics import score_image
    ref = list(ref_rows)
    out = []
    for t in thresholds:
        a_rows = []
        for r in ref:
            if r["image"] not in dets:
                continue
            detected = count_detections(dets[r["image"]], t)
            a_rows.append({"image": r["image"], "expected": r["expected"], "detected": detected,
                           "prompt_exact": score_image(r["expected"], detected).prompt_exact})
        stats = [pair_stats(a, b) for a, b in join(a_rows, ref)]
        s = _summary(stats, [1.0] * len(stats))
        out.append({"threshold": t, "n": len(stats), **s})
    return out
