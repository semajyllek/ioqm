import pytest

from ioqm.detect import count_detections, normalize_detections
from ioqm.metrics import bootstrap, paired_bootstrap_diff, score_image, summarize


def test_bug1_overcount_cannot_hide_undercount():
    # 2023 soft_ioqm gave 1.0 here
    s = score_image({"cat": 2, "dog": 2}, {"cat": 4, "dog": 0})
    assert s.type_soft == 0.0
    assert s.prompt_exact == 0.0 and s.type_exact == 0.0
    assert s.signed_err == 0.0 and s.abs_err == 2.0


def test_bug2_extra_classes_reported():
    s = score_image({"cat": 1}, {"cat": 1, "dog": 7, "person": 3})
    assert s.prompt_exact == 1.0  # the count score is unaffected...
    assert s.extra_class == 1.0   # ...but the extra classes are reported separately
    assert s.extra_classes == ("dog", "person")


def test_exact_and_partial():
    s = score_image({"cup": 3, "apple": 2, "dog": 1}, {"cup": 3, "apple": 3})
    assert s.prompt_exact == 0.0
    assert s.type_exact == pytest.approx(1 / 3)
    assert s.type_soft == pytest.approx((1 + 0.5 + 0) / 3)
    assert s.presence == pytest.approx(2 / 3)
    assert s.signed_err == pytest.approx((0 + 1 - 1) / 3)
    assert s.extra_class == 0.0


def test_perfect():
    s = score_image({"cup": 3}, {"cup": 3})
    assert (s.prompt_exact, s.type_exact, s.type_soft, s.abs_err) == (1.0, 1.0, 1.0, 0.0)


def test_soft_clipped_per_type():
    assert score_image({"cup": 1}, {"cup": 10}).type_soft == 0.0


def test_invalid_expected():
    with pytest.raises(ValueError):
        score_image({}, {"cup": 1})


def _rows(p_success, n_prompts=200, seeds=4):
    import random
    rng = random.Random(1)
    rows = []
    for p in range(n_prompts):
        for s in range(seeds):
            rows.append({"prompt_id": f"p{p}", "seed": s, "prompt_exact": float(rng.random() < p_success)})
    return rows


def test_bootstrap_ci_contains_mean():
    e = bootstrap(_rows(0.3), "prompt_exact", n_boot=500)
    assert e.lo <= e.mean <= e.hi
    assert e.n_prompts == 200 and e.n_images == 800
    assert 0.2 < e.mean < 0.4


def test_paired_diff_detects_difference():
    est, p = paired_bootstrap_diff(_rows(0.6), _rows(0.3), "prompt_exact", n_boot=500)
    assert est.lo > 0 and p < 0.01


def test_summarize_by_stratum():
    rows = [dict(r, n_types=1 + i % 2) for i, r in enumerate(_rows(0.5, 50, 2))]
    out = summarize(rows, by=lambda r: r["n_types"], metrics=("prompt_exact",), n_boot=100)
    assert set(out) == {1, 2}


def test_count_detections_thresholds_and_canonical():
    raw = [
        {"label": "person", "score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}},
        {"label": "person", "score": 0.4, "box": {}},
        {"label": "carrot", "score": 0.8, "box": {}},
        {"label": "N/A", "score": 0.99, "box": {}},
        {"label": "hat", "score": 0.99, "box": {}},
    ]
    dets = normalize_detections(raw)
    assert count_detections(dets, 0.5) == {"person": 1, "carrot": 1}
    assert count_detections(dets, 0.3) == {"person": 2, "carrot": 1}
    assert count_detections(dets, {"person": 0.3}, default=0.85) == {"person": 2}
    # with the canonical map, "carrot" never counts as "car"
    assert score_image({"car": 1}, count_detections(dets, 0.5)).presence == 0.0
