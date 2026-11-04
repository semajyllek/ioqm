import itertools
import math
import random

import pytest

from ioqm.analysis import dependence, features
from ioqm.best_of_n import selection_key
from ioqm.cooccurrence import build, pmi_fn
from ioqm.overopt import (bok_weights, bon_kl, check_independent, concrete_pick, curve, ensemble_rows,
                          expected_best_of_k, picks_queue)


def row(pid, seed, exact, soft=None, extra=0.0, expected=None, detected=None, split="test_id"):
    exp = expected or {"dog": 2}
    return {"image": f"{pid}_s{seed}.png", "prompt_id": pid, "seed": seed, "prompt_exact": float(exact),
            "type_exact": float(exact), "type_soft": float(exact if soft is None else soft),
            "abs_err": 0.0 if exact else 1.0, "extra_class": float(extra), "split": split,
            "expected": exp, "detected": detected if detected is not None else exp,
            "n_types": len(exp), "max_count": max(exp.values())}


def test_weights_sum_to_one_and_k1_uniform():
    for n in (1, 5, 64):
        for k in sorted({1, min(2, n), n}):
            assert sum(bok_weights(n, k)) == pytest.approx(1.0)
    assert bok_weights(4, 1) == [0.25] * 4
    assert bok_weights(4, 4) == [0, 0, 0, 1.0]


def test_estimator_matches_brute_force():
    rng = random.Random(3)
    n = 6
    proxy = [row("p", s, rng.random() < .5, soft=rng.random()) for s in range(n)]
    gold = [row("p", s, rng.random() < .5) for s in range(n)]
    rows, info = expected_best_of_k(proxy, gold, ks=range(1, n + 1))
    for k in range(1, n + 1):
        picks = [max(sub, key=selection_key) for sub in itertools.combinations(proxy, k)]
        brute = sum(gold[p["seed"]]["prompt_exact"] for p in picks) / len(picks)
        est = next(r for r in rows if r["k"] == k)["gold_prompt_exact"]
        assert est == pytest.approx(brute, abs=1e-12)
        brute_self = sum(p["prompt_exact"] for p in picks) / len(picks)
        assert next(r for r in rows if r["k"] == k)["proxy_self"] == pytest.approx(brute_self, abs=1e-12)


def test_oracle_dominates_and_gap_appears_with_misleading_proxy():
    rng = random.Random(0)
    proxy, gold = [], []
    for p in range(200):
        for s in range(16):
            truth = rng.random() < 0.3
            fooled = (not truth) and rng.random() < 0.25     # proxy says exact when it isn't
            proxy.append(row(f"p{p}", s, truth or fooled))
            gold.append(row(f"p{p}", s, truth))
    res, info = curve(proxy, gold, ks=(1, 4, 16), n_boot=200)
    assert info["prompts"] == 200
    assert res[16]["oracle"].mean >= res[16]["gold"].mean
    assert res[16]["gap"] > res[1]["gap"] > 0
    assert res[16]["gold"].mean > res[1]["gold"].mean


def test_missing_gold_and_small_prompts_counted():
    proxy = [row("a", s, 1) for s in range(4)] + [row("b", s, 1) for s in range(2)]
    gold = [row("a", s, 1) for s in range(3)] + [row("b", s, 1) for s in range(2)]
    rows, info = expected_best_of_k(proxy, gold, ks=(1, 3))
    assert info["images_without_gold"] == 1
    assert info["skipped_prompt_k"] == 1      # prompt b has 2 seeds < k=3


def test_ensemble_and_circularity_guard():
    a = [row("p", 0, 1, soft=1.0), row("p", 1, 1, soft=1.0)]
    b = [row("p", 0, 0, soft=0.5), row("p", 1, 1, soft=1.0)]
    ens = {r["seed"]: r for r in ensemble_rows([a, b])}
    assert ens[0]["prompt_exact"] == 0.0 and ens[1]["prompt_exact"] == 1.0
    assert ens[0]["type_soft"] == pytest.approx(0.75)
    check_independent(["yolov8x", "ens:yolov8x+mask2former"], "qwen3_vl_8b")
    with pytest.raises(ValueError):
        check_independent(["ens:yolov8x+qwen3_vl_8b"], "qwen3_vl_8b")


def test_kl():
    assert bon_kl(1) == 0.0
    assert bon_kl(64) == pytest.approx(math.log(64) - 63 / 64)


def test_concrete_pick_and_picks_queue():
    proxy = {"yolo": [row(f"p{p}", s, s == 2) for p in range(5) for s in range(4)]}
    gold = [row(f"p{p}", s, s == 3) for p in range(5) for s in range(4)]
    by = {r["seed"]: r for r in proxy["yolo"] if r["prompt_id"] == "p0"}
    assert concrete_pick(by, 1) == 0 and concrete_pick(by, 4) == 2
    q = picks_queue("R", proxy, gold, ks=(1, 4), n_prompts=3, seed=0)
    assert len({it["prompt_id"] for it in q}) == 3
    assert all(it["gold_prompt_exact"] == 0.0 for it in q)          # gold disagrees with every pick
    roles = [(r["proxy"], r["k"]) for it in q for r in it["roles"]]
    assert sorted(set(roles)) == [("yolo", 1), ("yolo", 4)] and len(roles) == 6


# ------------------------------------------------------------------ mechanism

def _two_type_rows(n_prompts, n_seeds, mode, rng):
    exp = {"cat": 2, "dog": 2}
    rows = []
    for p in range(n_prompts):
        for s in range(n_seeds):
            if mode == "independent":
                ec, ed = (0 if rng.random() < .5 else rng.choice([-1, 1]) for _ in range(2))
            elif mode == "clustered":       # all right or all wrong together
                ok = rng.random() < .5
                ec = ed = 0 if ok else -1
            else:                           # trading: total fixed, split varies
                ec = rng.choice([-1, 0, 1])
                ed = -ec
            det = {"cat": 2 + ec, "dog": 2 + ed}
            rows.append({**row(f"p{p}", s, ec == 0 and ed == 0, expected=exp, detected=det)})
    return rows


def test_dependence_detects_clustering_trading_and_independence():
    rng = random.Random(1)
    ind = dependence(_two_type_rows(60, 16, "independent", rng), n_perm=200)
    assert abs(ind["joint_minus_product"]) < 0.03 and ind["p_independence"] > 0.01
    clu = dependence(_two_type_rows(60, 16, "clustered", rng), n_perm=200)
    assert clu["joint_minus_product"] > 0.15 and clu["p_independence"] < 0.01
    trd = dependence(_two_type_rows(60, 16, "trading", rng), n_perm=200)
    assert trd["trade_corr"] < -0.9 and trd["p_trade"] < 0.01


def test_pmi_and_features():
    stats = build([{"cat", "dog"}, {"cat", "dog"}, {"cat"}, {"toaster"}])
    pmi = pmi_fn(stats)
    assert pmi(["cat", "dog"]) > pmi(["cat", "toaster"])
    assert pmi(["cat", "dog", "toaster"]) == pytest.approx(min(pmi(["cat", "toaster"]), pmi(["dog", "toaster"])))
    f = features(row("p", 0, 1, expected={"cat": 1, "dog": 3}, split="test_ood_pair"), pmi)
    assert f["n_types"] == 2 and f["max_count"] == 3 and f["total_instances"] == 4 and f["held_out"] == 1.0
