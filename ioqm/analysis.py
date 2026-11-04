"""
Mechanism analyses on score rows (no GPU). Run them on many-seed runs.

1. Independence of per-type errors, within prompt.
   For a prompt with types A, B (, C) and n seeds, estimate p_A = P(A exact),
   p_B, ... and the observed joint P(all exact). Under independence the joint
   equals the product of the marginals. Averaging (joint - product) over
   prompts controls for everything about the prompt. Significance comes from a
   permutation null: shuffle each type's outcomes across seeds independently,
   within prompt, which keeps every marginal and breaks only the dependence.
     joint > product  -> errors cluster: images tend to be all right or all wrong
     joint < product  -> errors repel

2. Trading between types, within prompt.
   Mean over prompts of the correlation between two types' signed count errors
   across seeds (same permutation null).
     negative -> the model trades instances between classes (gets the total
                 roughly right but splits it wrong)
     positive -> global over- or under-generation that hits all types together

3. What predicts failure.
   Logistic regression of prompt_exact on prompt features, standard errors
   clustered by prompt (statsmodels). Features: number of types, largest count,
   total instances, held-out pair/triple, and (optional) the minimum pairwise
   COCO PMI of the class set.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

HELD_OUT_SPLITS = ("test_ood_pair", "test_ood_triple")


def _group(rows: Sequence[Mapping]) -> Dict[str, List[Mapping]]:
    g: Dict[str, List[Mapping]] = {}
    for r in rows:
        g.setdefault(r["prompt_id"], []).append(r)
    return g


def _type_matrix(rows: Sequence[Mapping]) -> Tuple[List[str], List[List[int]], List[List[int]]]:
    """classes, exact[seed][type] in {0,1}, signed error[seed][type]."""
    classes = list(rows[0]["expected"])
    exact, err = [], []
    for r in sorted(rows, key=lambda r: r["seed"]):
        e = [int(r["detected"].get(c, 0)) - int(r["expected"][c]) for c in classes]
        err.append(e)
        exact.append([int(x == 0) for x in e])
    return classes, exact, err


def _indep_stat(exact: List[List[int]]) -> Tuple[float, float]:
    n, t = len(exact), len(exact[0])
    joint = sum(all(row) for row in exact) / n
    prod = 1.0
    for j in range(t):
        prod *= sum(row[j] for row in exact) / n
    return joint, prod


def _corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _trade_stat(err: List[List[int]]) -> Optional[float]:
    t = len(err[0])
    cs = [c for i in range(t) for j in range(i + 1, t)
          if (c := _corr([e[i] for e in err], [e[j] for e in err])) is not None]
    return sum(cs) / len(cs) if cs else None


def dependence(rows: Sequence[Mapping], min_seeds: int = 8, n_perm: int = 500, seed: int = 0) -> dict:
    """Independence and trading statistics over prompts with >= 2 types and >= min_seeds seeds."""
    prompts = []
    for pid, rs in _group(rows).items():
        if len(rs[0]["expected"]) >= 2 and len(rs) >= min_seeds:
            prompts.append(_type_matrix(rs))
    if not prompts:
        raise ValueError(f"no prompts with >= 2 types and >= {min_seeds} seeds")

    def stats(mats_exact, mats_err):
        diffs, joints, prods, trades = [], [], [], []
        for ex, er in zip(mats_exact, mats_err):
            j, p = _indep_stat(ex)
            diffs.append(j - p); joints.append(j); prods.append(p)
            t = _trade_stat(er)
            if t is not None:
                trades.append(t)
        return (sum(diffs) / len(diffs), sum(joints) / len(joints), sum(prods) / len(prods),
                sum(trades) / len(trades) if trades else math.nan, len(trades))

    ex = [p[1] for p in prompts]
    er = [p[2] for p in prompts]
    d_obs, joint, prod, t_obs, n_trade = stats(ex, er)
    rng = random.Random(seed)
    d_null, t_null = [], []
    for _ in range(n_perm):
        # one shared permutation per prompt keeps exact and error aligned
        ex_p, er_p = [], []
        for e1, e2 in zip(ex, er):
            idx = [list(range(len(e1))) for _ in range(len(e1[0]))]
            for ix in idx:
                rng.shuffle(ix)
            ex_p.append([[e1[idx[j][s]][j] for j in range(len(e1[0]))] for s in range(len(e1))])
            er_p.append([[e2[idx[j][s]][j] for j in range(len(e2[0]))] for s in range(len(e2))])
        dn, _, _, tn, _ = stats(ex_p, er_p)
        d_null.append(dn)
        t_null.append(tn)

    def p_two_sided(obs, null):
        null = [x for x in null if not math.isnan(x)]
        if not null or math.isnan(obs):
            return math.nan
        c = sum(abs(x) >= abs(obs) for x in null)
        return (c + 1) / (len(null) + 1)

    return {
        "n_prompts": len(prompts),
        "joint": joint, "product": prod, "joint_minus_product": d_obs,
        "joint_over_product": joint / prod if prod else math.nan,
        "p_independence": p_two_sided(d_obs, d_null),
        "null_mean_diff": sum(d_null) / len(d_null),
        "trade_corr": t_obs, "n_prompts_trade": n_trade,
        "p_trade": p_two_sided(t_obs, t_null),
        "null_mean_trade": sum(x for x in t_null if not math.isnan(x)) / max(1, sum(not math.isnan(x) for x in t_null)),
    }


def features(row: Mapping, pmi=None) -> dict:
    exp = row["expected"]
    f = {
        "n_types": len(exp),
        "max_count": max(exp.values()),
        "total_instances": sum(exp.values()),
        "held_out": float(row.get("split") in HELD_OUT_SPLITS),
    }
    if pmi is not None:
        f["min_pmi"] = pmi(list(exp)) if len(exp) > 1 else math.nan
    return f


def failure_regression(rows: Sequence[Mapping], pmi=None, multi_type_only: bool = True):
    """statsmodels GLM (binomial) of prompt_exact on prompt features, SEs clustered by prompt.
    Returns (fitted result, feature names). Features are z-scored, so coefficients are
    log-odds per standard deviation."""
    import numpy as np
    import statsmodels.api as sm

    use = [r for r in rows if not multi_type_only or len(r["expected"]) >= 2]
    feats = [features(r, pmi) for r in use]
    names = [k for k in feats[0] if not all(math.isnan(f[k]) for f in feats)]
    keep = [i for i, f in enumerate(feats) if not any(math.isnan(f[k]) for k in names)]
    X = np.array([[feats[i][k] for k in names] for i in keep], dtype=float)
    sd = X.std(axis=0)
    names = [n for n, s in zip(names, sd) if s > 0]
    X = X[:, sd > 0]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # drop features that add no rank (e.g. total_instances when it is determined by the others)
    cols, dropped = [], []
    for j, n in enumerate(names):
        trial = cols + [j]
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X[:, trial]])) == len(trial) + 1:
            cols = trial
        else:
            dropped.append(n)
    X, names = X[:, cols], [names[j] for j in cols]
    y = np.array([float(use[i]["prompt_exact"]) for i in keep])
    groups = np.unique([use[i]["prompt_id"] for i in keep], return_inverse=True)[1]
    model = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    res.dropped_features = dropped
    return res, ["const"] + names
