import random

import pytest

from ioqm.best_of_n import curve, select, selection_key
from ioqm.counters import DEFAULT_THRESHOLDS, LEGACY_ALIASES, resolve


def row(pid, seed, exact, soft=None, extra=0.0):
    return {"prompt_id": pid, "seed": seed, "prompt_exact": float(exact),
            "type_exact": float(exact), "type_soft": float(exact if soft is None else soft),
            "abs_err": 0.0 if exact else 1.0, "signed_err": 0.0, "extra_class": float(extra),
            "split": "test_id", "n_types": 2, "max_count": 2}


def test_selection_key_prefers_exact_then_soft_then_no_extra_then_low_seed():
    rows = [row("p", 0, 0, 0.9), row("p", 1, 1, 1.0, extra=1), row("p", 2, 1, 1.0), row("p", 3, 1, 1.0)]
    assert max(rows, key=selection_key)["seed"] == 2


def test_groups_are_disjoint_and_k1_is_every_image():
    v = [row("p", s, s == 5) for s in range(8)]
    e = [row("p", s, s == 5) for s in range(8)]
    assert len(select(v, e, 1)) == 8
    k4 = select(v, e, 4)
    assert [r["group"] for r in k4] == [0, 1]
    assert [r["prompt_exact"] for r in k4] == [0.0, 1.0]  # the correct image (seed 5) is in group 1 only


def test_verifier_and_evaluator_are_separate():
    # verifier wrongly likes seed 0; evaluator says only seed 1 is right
    v = [row("p", 0, 1), row("p", 1, 0)]
    e = [row("p", 0, 0), row("p", 1, 1)]
    (r,) = select(v, e, 2)
    assert r["seed"] == 0
    assert r["prompt_exact"] == 0.0                 # the evaluator's verdict on the pick
    assert r["verifier_self_prompt_exact"] == 1.0   # optimistic self-score
    assert r["oracle_prompt_exact"] == 1.0          # pass@2 under the evaluator


def test_only_images_scored_by_both_counters_are_used():
    v = [row("p", s, 1) for s in range(4)]
    e = [row("p", s, 1) for s in range(3)]
    assert len(select(v, e, 1)) == 3


def test_curve_gain_with_informative_verifier():
    rng = random.Random(0)
    v, e = [], []
    for p in range(150):
        for s in range(8):
            truth = rng.random() < 0.3
            noisy = truth if rng.random() < 0.85 else not truth   # 85%-accurate verifier
            e.append(row(f"p{p}", s, truth))
            v.append(row(f"p{p}", s, noisy))
    res = curve(v, e, ks=(1, 2, 4, 8), n_boot=300)
    assert 0.25 < res[1]["selected"].mean < 0.35
    assert res[8]["selected"].mean > res[1]["selected"].mean + 0.2
    assert res[8]["gain_vs_k1"].lo > 0
    assert res[8]["oracle"].mean >= res[8]["selected"].mean


def test_counter_registry():
    assert resolve("yolo") == "yolov8x" and resolve("mask2former") == "mask2former"
    assert set(LEGACY_ALIASES.values()) <= set(DEFAULT_THRESHOLDS)
    with pytest.raises(KeyError):
        resolve("nope")
