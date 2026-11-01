"""
IOQM v3 metrics. These are fixed in advance of any results (see docs/plan.md, Phase 0).

All inputs are keyed by canonical COCO names (see ioqm.classes). `expected`
holds the requested counts; `detected` holds the counts of every COCO-80 class
found in the image.

Per image:
  prompt_exact  1 if every requested type has exactly the requested count. PRIMARY.
  type_exact    fraction of requested types with the exact count (fixed hard IOQM).
  type_soft     mean over types of 1 - min(1, |det - exp| / exp). Clipped per
                type, so overcounting one type cannot hide undercounting another.
  abs_err       mean over types of |det - exp|.
  signed_err    mean over types of (det - exp). Positive means overgeneration.
  presence      fraction of requested types detected at least once.
  extra_class   1 if any COCO-80 class that was NOT requested is detected.
                Reported separately, never folded into the count scores.

Aggregation uses a cluster bootstrap over prompts: all seeds of one prompt are
resampled together. Comparing two systems uses a paired bootstrap on the same
prompts and seeds.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

METRIC_NAMES = ("prompt_exact", "type_exact", "type_soft", "abs_err", "signed_err", "presence", "extra_class")


@dataclass(frozen=True)
class ImageScores:
    prompt_exact: float
    type_exact: float
    type_soft: float
    abs_err: float
    signed_err: float
    presence: float
    extra_class: float
    extra_classes: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["extra_classes"] = list(self.extra_classes)
        return d


def score_image(expected: Mapping[str, int], detected: Mapping[str, int]) -> ImageScores:
    if not expected:
        raise ValueError("expected must contain at least one class")
    if any(c < 1 for c in expected.values()):
        raise ValueError(f"expected counts must be >= 1: {dict(expected)}")
    n = len(expected)
    exact = soft = abs_e = signed = present = 0.0
    for name, e in expected.items():
        d = int(detected.get(name, 0))
        exact += d == e
        soft += 1.0 - min(1.0, abs(d - e) / e)
        abs_e += abs(d - e)
        signed += d - e
        present += d > 0
    extras = tuple(sorted(k for k, v in detected.items() if v > 0 and k not in expected))
    return ImageScores(
        prompt_exact=float(exact == n),
        type_exact=exact / n,
        type_soft=soft / n,
        abs_err=abs_e / n,
        signed_err=signed / n,
        presence=present / n,
        extra_class=float(bool(extras)),
        extra_classes=extras,
    )


# --------------------------------------------------------------- aggregation

@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n_prompts: int
    n_images: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}] (prompts={self.n_prompts}, images={self.n_images})"


def _group(rows: Sequence[Mapping], cluster_key: str) -> Dict[Hashable, List[Mapping]]:
    g: Dict[Hashable, List[Mapping]] = {}
    for r in rows:
        g.setdefault(r[cluster_key], []).append(r)
    return g


def _cluster_means(groups: Dict[Hashable, List[Mapping]], metric: str) -> Dict[Hashable, Tuple[float, int]]:
    # (sum, n_images) for each cluster, so that the pooled mean weights images equally
    return {k: (sum(float(r[metric]) for r in v), len(v)) for k, v in groups.items()}


def _quantile(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return math.nan
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap(
    rows: Sequence[Mapping],
    metric: str,
    cluster_key: str = "prompt_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Mean of `metric` over images, with a percentile CI from resampling prompts."""
    groups = _group(rows, cluster_key)
    if not groups:
        raise ValueError("no rows")
    stats = _cluster_means(groups, metric)
    keys = list(stats)
    tot_s = sum(s for s, _ in stats.values())
    tot_n = sum(n for _, n in stats.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = n = 0.0
        for _ in keys:
            cs, cn = stats[keys[rng.randrange(len(keys))]]
            s += cs
            n += cn
        boots.append(s / n)
    return Estimate(tot_s / tot_n, _quantile(boots, alpha / 2), _quantile(boots, 1 - alpha / 2), len(keys), int(tot_n))


def paired_bootstrap_diff(
    rows_a: Sequence[Mapping],
    rows_b: Sequence[Mapping],
    metric: str,
    cluster_key: str = "prompt_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[Estimate, float]:
    """
    Difference A - B on the prompts both systems share. Returns the estimate
    and a two-sided bootstrap p-value (2 * min share of resamples on either
    side of 0).
    """
    ga, gb = _group(rows_a, cluster_key), _group(rows_b, cluster_key)
    shared = sorted(set(ga) & set(gb), key=str)
    if not shared:
        raise ValueError("no shared prompts")
    sa = _cluster_means({k: ga[k] for k in shared}, metric)
    sb = _cluster_means({k: gb[k] for k in shared}, metric)

    def diff(idx: Iterable[Hashable]) -> float:
        a_s = a_n = b_s = b_n = 0.0
        for k in idx:
            a_s += sa[k][0]; a_n += sa[k][1]
            b_s += sb[k][0]; b_n += sb[k][1]
        return a_s / a_n - b_s / b_n

    point = diff(shared)
    rng = random.Random(seed)
    boots = [diff(shared[rng.randrange(len(shared))] for _ in shared) for _ in range(n_boot)]
    p = 2 * min(sum(b <= 0 for b in boots), sum(b >= 0 for b in boots)) / n_boot
    n_img = int(sum(v[1] for v in sa.values()) + sum(v[1] for v in sb.values()))
    est = Estimate(point, _quantile(boots, alpha / 2), _quantile(boots, 1 - alpha / 2), len(shared), n_img)
    return est, min(1.0, p)


def summarize(
    rows: Sequence[Mapping],
    by: Optional[Callable[[Mapping], Hashable]] = None,
    metrics: Sequence[str] = METRIC_NAMES,
    **kw,
) -> Dict[Hashable, Dict[str, Estimate]]:
    """Bootstrap estimates for each metric, overall or per stratum (`by` maps a row to a stratum key)."""
    strata: Dict[Hashable, List[Mapping]] = {}
    for r in rows:
        strata.setdefault(by(r) if by else "all", []).append(r)
    return {k: {m: bootstrap(v, m, **kw) for m in metrics} for k, v in sorted(strata.items(), key=lambda x: str(x[0]))}
