import json
import sys
from pathlib import Path

import pytest

from ioqm.agreement import agreement, pair_stats, threshold_sweep
from ioqm.counters import PARSE_ERROR, parse_vlm_counts, vlm_question
from ioqm.detect import detect_run, score_run
from ioqm.labeler import add_point, make_record, next_index, point_counts, undo
from ioqm.labels import (HUMAN, dedupe, labels_by_run, parse_legacy_label_file, read_labels,
                         write_human_detections)
from ioqm.sampling import (_allocate, interleave, item_id, labeled_weights, sample_run, stratum)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def srow(image, expected, detected, pid=None):
    exact = all(detected.get(c, 0) == n for c, n in expected.items())
    return {"image": image, "prompt_id": pid or image, "expected": expected, "detected": detected,
            "prompt_exact": float(exact), "n_types": len(expected), "max_count": max(expected.values())}


# ------------------------------------------------------------------ labels

def test_legacy_label_parsing(tmp_path):
    p = tmp_path / "v2_stable-diffusion-v1-5_images_human_labels.jsonl"
    lines = [
        {"2_broccolis_and_1_toaster.png": {"broccolis": 2, "toaster": 1.0}},
        {"3_mouses.png": {"mouses": 2}},
        {"1_cat_and_2_dogs.png": {"cat": None, "dogs": 2}},          # missing count -> unusable
        {"2_broccolis_and_1_toaster.png": {"broccolis": 3, "toaster": 1}},   # relabel
    ]
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    good, bad = parse_legacy_label_file(p)
    assert good[0] == ("2_broccolis_and_1_toaster.png", {"broccoli": 2, "toaster": 1})
    assert good[1] == ("3_mouses.png", {"mouse": 2})
    assert len(bad) == 1 and "missing" in bad[0][1]
    last, repeats = dedupe(good)
    assert last["2_broccolis_and_1_toaster.png"] == {"broccoli": 3, "toaster": 1}
    assert repeats == [("2_broccolis_and_1_toaster.png", {"broccoli": 2, "toaster": 1},
                        {"broccoli": 3, "toaster": 1})]


def test_human_labels_score_like_a_counter(tmp_path):
    from human_check import build_run
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for name in ("2_dogs.png", "1_cat_and_2_dogs.png"):
        (imgs / name).write_bytes(b"x")
    run = tmp_path / "run"
    n = build_run(run, imgs, {"2_dogs.png": {"dog": 2}, "1_cat_and_2_dogs.png": {"cat": 1, "dog": 1},
                              "missing.png": {"dog": 1}}, "v2", "sd15")
    assert n == 2
    rows = {r["image"]: r for r in score_run(run, HUMAN, 0.5, "t0.5")}
    assert rows["2_dogs.png"]["prompt_exact"] == 1.0
    assert rows["1_cat_and_2_dogs.png"]["prompt_exact"] == 0.0
    assert rows["1_cat_and_2_dogs.png"]["signed_err"] == -0.5


def test_new_label_format_last_wins_and_usable_filter(tmp_path):
    p = tmp_path / "labels.jsonl"
    item = {"item_id": "a", "run_dir": "R", "image": "x.png"}
    recs = [make_record(item, {"dog": 1}, False, False, "", {}, "me"),
            make_record(item, {"dog": 2}, False, False, "", {}, "me"),
            make_record({**item, "item_id": "b", "image": "y.png"}, {"dog": 1}, True, False, "", {}, "me"),
            make_record({**item, "item_id": "c", "image": "z.png"}, None, False, True, "", {}, "me")]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    latest = read_labels(p)
    assert latest["a"]["counts"] == {"dog": 2}
    assert labels_by_run(latest.values()) == {"R": {"x.png": {"dog": 2}}}   # ambiguous + skipped dropped


# ------------------------------------------------------------------ agreement

def test_agreement_precision_kappa():
    exp = {"dog": 2}
    ref = [srow(f"i{k}", exp, {"dog": 2 if k < 5 else 1}) for k in range(10)]   # truth: 5 exact
    # counter says exact on i0..i7: 5 true positives, 3 false positives
    cnt = [srow(f"i{k}", exp, {"dog": 2 if k < 8 else 3}) for k in range(10)]
    res = agreement(cnt, ref, n_boot=50)
    assert res["precision"][0] == pytest.approx(5 / 8)
    assert res["recall"][0] == pytest.approx(1.0)
    assert res["accuracy"][0] == pytest.approx(0.7)
    assert res["kappa"][0] == pytest.approx((0.7 - 0.5) / 0.5)
    assert res["counter_rate"][0] == pytest.approx(0.8) and res["reference_rate"][0] == pytest.approx(0.5)


def test_agreement_weights_change_estimate():
    exp = {"dog": 1}
    ref = [srow("a", exp, {"dog": 1}), srow("b", exp, {"dog": 0})]
    cnt = [srow("a", exp, {"dog": 1}), srow("b", exp, {"dog": 1})]
    assert agreement(cnt, ref, n_boot=10)["accuracy"][0] == pytest.approx(0.5)
    assert agreement(cnt, ref, {"a": 9.0, "b": 1.0}, n_boot=10)["accuracy"][0] == pytest.approx(0.9)


def test_pair_stats_signed():
    s = pair_stats(srow("a", {"cat": 1, "dog": 2}, {"cat": 0, "dog": 4}),
                   srow("a", {"cat": 1, "dog": 2}, {"cat": 1, "dog": 2}))
    assert s["signed_diff"] == pytest.approx(0.5) and s["count_agree"] == 0.0


def test_threshold_sweep_finds_right_threshold():
    exp = {"dog": 2}
    ref = [srow(f"i{k}", exp, {"dog": 2}) for k in range(4)]
    dets = {f"i{k}": [{"label": "dog", "score": 0.9}, {"label": "dog", "score": 0.7},
                      {"label": "dog", "score": 0.3}] for k in range(4)}
    res = {r["threshold"]: r for r in threshold_sweep(dets, ref, (0.2, 0.5, 0.8))}
    assert res[0.5]["accuracy"] == 1.0 and res[0.2]["accuracy"] == 0.0 and res[0.8]["accuracy"] == 0.0


# ------------------------------------------------------------------ sampling

def test_strata():
    e = {"cat": 1, "dog": 2}
    ok = {"cat": 1, "dog": 2}
    assert stratum(srow("a", e, ok), srow("a", e, {"cat": 1})) == "disagree"
    assert stratum(srow("a", e, ok), srow("a", e, ok)) == "both_exact"
    assert stratum(srow("a", e, {"cat": 0, "dog": 2}), srow("a", e, {"dog": 2})) == "both_wrong_same"
    assert stratum(srow("a", e, {"cat": 0, "dog": 2}), srow("a", e, {"dog": 3})) == "both_wrong_diff"


def test_allocation_redistributes_when_a_stratum_is_small():
    got = _allocate(100, {"disagree": .5, "both_exact": .5}, {"disagree": 10, "both_exact": 500})
    assert got["disagree"] == 10 and sum(got.values()) == 100


def _two_counters(n=200):
    e = {"dog": 2}
    a = [srow(f"im{k}", e, {"dog": 2 if k % 2 else 1}, pid=f"p{k // 4}") for k in range(n)]
    b = [srow(f"im{k}", e, {"dog": 2 if k % 3 else 1}, pid=f"p{k // 4}") for k in range(n)]
    return a, b


def test_sample_run_weights_deterministic_and_exclude():
    a, b = _two_counters()
    items = sample_run("R", a, b, 40, "yolo", "m2f", seed=1)
    assert len(items) == 40
    assert items == sample_run("R", a, b, 40, "yolo", "m2f", seed=1)
    # weights sum to the population of each sampled stratum
    tot = {}
    for it in items:
        tot.setdefault(it["stratum"], 0.0)
        tot[it["stratum"]] += it["weight"]
    for s, w in tot.items():
        assert w == pytest.approx(next(it["stratum_size"] for it in items if it["stratum"] == s))
    assert "expected" in items[0] and items[0]["classes"] == ["dog"]
    again = sample_run("R", a, b, 40, "yolo", "m2f", seed=1, exclude=[it["item_id"] for it in items])
    assert not {it["item_id"] for it in items} & {it["item_id"] for it in again}


def test_sample_prefers_distinct_prompts():
    a, b = _two_counters()
    items = sample_run("R", a, b, 20, "yolo", "m2f", seed=0)
    for s in {it["stratum"] for it in items}:
        pids = [it["prompt_id"] for it in items if it["stratum"] == s]
        assert len(set(pids)) == len(pids)


def test_labeled_weights_shrink_with_skips():
    a, b = _two_counters()
    items = sample_run("R", a, b, 40, "yolo", "m2f", seed=1)
    s0 = items[0]["stratum"]
    in_s = [it for it in items if it["stratum"] == s0]
    keep = [it["image"] for it in in_s[: max(1, len(in_s) // 2)]]
    w = labeled_weights(items, "R", keep)
    assert sum(w.values()) == pytest.approx(in_s[0]["stratum_size"])


def test_interleave_keeps_everything():
    q = interleave({"A": [{"item_id": str(i)} for i in range(5)], "B": [{"item_id": "b"}]}, seed=0)
    assert sorted(x["item_id"] for x in q) == ["0", "1", "2", "3", "4", "b"]


# ------------------------------------------------------------------ labeler + VLM

def test_labeler_click_logic():
    pts, hist = {}, []
    add_point(pts, hist, "cat", (10, 20))
    add_point(pts, hist, "dog", (5, 5))
    add_point(pts, hist, "cat", (30, 40))
    assert point_counts(["cat", "dog"], pts) == [2, 1]
    undo(pts, hist)
    assert point_counts(["cat", "dog"], pts) == [1, 1]
    q = [{"item_id": i} for i in "abcd"]
    assert next_index(q, ["a", "b"]) == 2
    assert next_index(q, ["a", "c", "d"], start=3) == 1   # wraps around
    assert next_index(q, list("abcd")) is None


def test_vlm_parse():
    q, keys = vlm_question(["cat", "person"])
    assert '"people"' in q
    assert parse_vlm_counts('Sure: {"cats": 2, "people": 0}', keys) == {"cat": 2, "person": 0}
    assert parse_vlm_counts('{"cats": "two", "people": 0}', keys) is None
    assert parse_vlm_counts('{"cats": 2}', keys) is None            # missing class
    assert parse_vlm_counts("no json here", keys) is None


def test_parse_error_images_are_excluded(tmp_path):
    from ioqm.generate import run_generation
    from ioqm.prompts import make_spec

    class FB:
        from ioqm.generate import REGISTRY
        config = REGISTRY["sd15"]

        def generate(self, prompt, seed):
            class Img:
                def save(self, p):
                    Path(p).write_text(prompt)
            return Img()

    specs = [make_spec([("dog", 2)]), make_spec([("cat", 1)])]
    run_generation(FB(), specs, [0], tmp_path, log_every=0)

    def fake_vlm(path, threshold, classes=None):
        if "dog" in classes:
            return [{"label": PARSE_ERROR, "score": 0.0, "box": {}}]
        return [{"label": "cat", "score": 1.0, "box": {}}]

    detect_run(tmp_path, fake_vlm, "vlm")
    rows = score_run(tmp_path, "vlm", 0.5, "t0.5", save=False)
    assert len(rows) == 1 and rows[0]["expected"] == {"cat": 1}


def test_detect_run_only_subset(tmp_path):
    from ioqm.generate import REGISTRY, read_manifest, run_generation
    from ioqm.prompts import make_spec

    class FB:
        config = REGISTRY["sd15"]

        def generate(self, prompt, seed):
            class Img:
                def save(self, p):
                    Path(p).write_text(prompt)
            return Img()

    run_generation(FB(), [make_spec([("dog", 2)]), make_spec([("cat", 1)])], [0, 1], tmp_path, log_every=0)
    names = [r["image"] for r in read_manifest(tmp_path)]
    det = lambda p, threshold, classes=None: []
    assert detect_run(tmp_path, det, "d", only=names[:1]) == 1
    assert detect_run(tmp_path, det, "d") == 3          # the rest, later


# ------------------------------------------------------------------ annotators and re-checks

def _write(path, recs):
    import json as _json
    with open(path, "w") as f:
        for r in recs:
            f.write(_json.dumps(r) + "\n")


def test_label_files_per_annotator_and_review(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("label_cli", Path(__file__).resolve().parents[1] / "scripts" / "label.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.labels_file(tmp_path, "v1", "labeler_1", False).name == "labels.jsonl"
    assert cli.labels_file(tmp_path, "picks", "labeler_2", False).name == "labels_picks__labeler_2.jsonl"
    assert cli.labels_file(tmp_path, "v1", "labeler_1", True).name == "labels__review.jsonl"


def test_merge_review_overrides_and_needs_review(tmp_path):
    from ioqm.labels import labelers_in, needs_review, read_label_files
    a = tmp_path / "a.jsonl"
    r = tmp_path / "r.jsonl"
    _write(a, [{"item_id": "x", "counts": {"dog": 1}, "ambiguous": True, "notes": "", "labeler": "labeler_1"},
               {"item_id": "y", "counts": {"dog": 2}, "ambiguous": False, "notes": "maybe 3", "labeler": "labeler_1"},
               {"item_id": "z", "counts": {"dog": 2}, "ambiguous": False, "notes": " ", "labeler": "labeler_1"}])
    _write(r, [{"item_id": "x", "counts": {"dog": 2}, "ambiguous": False, "notes": "", "labeler": "labeler_1"}])
    m = read_label_files([a, r])
    assert m["x"]["counts"] == {"dog": 2} and not m["x"]["ambiguous"]
    base = read_label_files([a])
    assert [k for k in "xyz" if needs_review(base[k])] == ["x", "y"]
    assert labelers_in(a) == {"labeler_1"}
