from pathlib import Path

import pytest

from ioqm.classes import COCO80, canonical_counts, canonicalize, prompt_noun
from ioqm.prompts import (
    build_v3, legacy_specs, load_specs, make_spec, parse_prompt, render, save_specs, stratified_subset,
)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ classes

def test_80_classes_contiguous_ids():
    assert len(COCO80) == 80
    assert [c.coco_id for c in COCO80] == list(range(80))


def test_class_names_match_yolo_yaml():
    yml = (REPO / "scripts" / "coco_classes.yml").read_text().splitlines()
    names = [l.split(":", 1)[1].strip() for l in yml if l.strip()[:1].isdigit()]
    assert names == [c.name for c in COCO80]


@pytest.mark.parametrize("surface,name", [
    ("people", "person"), ("person", "person"),          # bug 3a: people never matched person
    ("carrot", "carrot"), ("car", "car"), ("cars", "car"), ("carrots", "carrot"),  # bug 3b: car/carrot
    ("mouses", "mouse"), ("sheeps", "sheep"), ("broccolis", "broccoli"),            # bug 4: bad plurals
    ("ski", "skis"), ("ty", "tie"), ("ties", "tie"), ("tvs", "tv"), ("TV", "tv"),
    ("hair driers", "hair drier"), ("hair dryer", "hair drier"), ("wine glasses", "wine glass"),
    ("motorbike", "motorcycle"), ("sofa", "couch"),
])
def test_canonicalize(surface, name):
    assert canonicalize(surface) == name


def test_non_coco80_detector_labels_dropped():
    for lbl in ("N/A", "hat", "plate", "street sign", "window"):
        assert canonicalize(lbl) is None


def test_prompt_nouns_hand_written():
    assert prompt_noun("person", 2) == "people"
    assert prompt_noun("sheep", 3) == "sheep"
    assert prompt_noun("mouse", 2) == "computer mice"
    assert prompt_noun("knife", 2) == "knives"
    assert prompt_noun("bus", 1) == "bus"


def test_canonical_counts_merges_aliases():
    assert canonical_counts({"people": 2, "person": 1, "cups": 3}) == {"person": 3, "cup": 3}


# ------------------------------------------------------------- parse/render

@pytest.mark.parametrize("text,expected", [
    ("5 wine glasses and 5 cups and 4 umbrellas", {"wine glass": 5, "cup": 5, "umbrella": 4}),
    ("3_broccolis_and_1_toaster", {"broccoli": 3, "toaster": 1}),
    ("2 people and 1 car on a table.", {"person": 2, "car": 1}),
    ("1 car and 2 carrots", {"car": 1, "carrot": 2}),
    ("2 hot dogs, 3 dogs", {"hot dog": 2, "dog": 3}),
    ("three teddy bears and a photo", {"teddy bear": 3}),
])
def test_parse_prompt(text, expected):
    assert parse_prompt(text) == expected


def test_parse_rejects_unknown():
    with pytest.raises(ValueError):
        parse_prompt("3 dragons")


@pytest.mark.parametrize("fname,n", [("prompts_v1.txt", 16175), ("prompts_v2.txt", 4005)])
def test_every_legacy_prompt_parses(fname, n):
    p = REPO / "prompts" / fname
    if not p.exists():
        pytest.skip("legacy prompt file not present (gitignored)")
    specs = legacy_specs(p.read_text().splitlines(), dataset=fname)
    assert len(specs) == n
    # every legacy prompt with "people" now maps to person
    people = [s for s in specs if "person" in s.expected and s.expected["person"] > 1]
    if fname == "prompts_v2.txt":
        assert len(people) > 0


def test_render_roundtrip():
    s = make_spec([("person", 2), ("cup", 1), ("mouse", 3)])
    assert render(s) == "2 people and 1 cup and 3 computer mice"
    assert render(s, "a photo of {objects}", ", ", "words") == "a photo of two people, one cup, three computer mice"
    for t in (render(s), render(s, numerals="words"), render(s, "{objects} on a table.", ", ")):
        assert parse_prompt(t) == s.expected


def test_spec_validation():
    with pytest.raises(ValueError):
        make_spec([("dog", 2), ("dogs", 1)])
    with pytest.raises(ValueError):
        make_spec([("dog", 0)])
    assert make_spec([("dogs", 2)]).prompt_id == make_spec([("dog", 2)]).prompt_id


# --------------------------------------------------------------- v3 builder

PAIR = ["person", "bicycle", "car", "bus", "boat", "dog", "cat", "horse", "sheep", "elephant",
        "bird", "umbrella", "backpack", "suitcase", "bottle", "wine glass", "cup", "bowl",
        "banana", "apple", "chair", "potted plant", "teddy bear", "clock"]
TRIPLE = ["person", "dog", "car", "cup", "apple", "elephant", "umbrella", "teddy bear"]


@pytest.fixture(scope="module")
def v3():
    return build_v3(PAIR, TRIPLE, seed=0)


def test_v3_sizes(v3):
    by = {k: sum(1 for s in v3 if s.n_types == k) for k in (1, 2, 3)}
    assert by == {1: 24 * 5, 2: 276 * 25, 3: 56 * 125}
    assert len({s.prompt_id for s in v3}) == len(v3)


def _pairs(cset):
    from itertools import combinations
    return {tuple(sorted(p)) for p in combinations(cset, 2)}


def test_v3_no_heldout_pair_leaks_into_train(v3):
    held_pairs = {s.class_set for s in v3 if s.split == "test_ood_pair" and s.n_types == 2}
    assert held_pairs
    train_pairs = set().union(*(_pairs(s.class_set) for s in v3 if s.split == "train"))
    assert not (held_pairs & train_pairs)  # not even inside train triples
    # every test_ood_pair prompt contains at least one held-out pair
    assert all(_pairs(s.class_set) & held_pairs for s in v3 if s.split == "test_ood_pair")


def test_v3_ood_triples_are_novel_combinations_of_seen_pairs(v3):
    train_sets = {s.class_set for s in v3 if s.split == "train"}
    train_pairs = set().union(*(_pairs(c) for c in train_sets))
    ood_triples = {s.class_set for s in v3 if s.split == "test_ood_triple"}
    assert ood_triples and not (ood_triples & train_sets)
    assert all(_pairs(t) <= train_pairs for t in ood_triples)


def test_v3_count_and_id_splits(v3):
    assert not any(sum(c == 5 for _, c in s.objects) >= 2 for s in v3 if s.split == "train")
    test_id_multi = {s.class_set for s in v3 if s.split == "test_id" and s.n_types > 1}
    assert test_id_multi <= {s.class_set for s in v3 if s.split == "train"}
    assert all(s.split == "test_single" for s in v3 if s.n_types == 1)
    triples_in_train = {s.class_set for s in v3 if s.split == "train" and s.n_types == 3}
    assert len(triples_in_train) >= 20  # enough triple class sets left for RL training


def test_v3_deterministic(v3):
    again = build_v3(PAIR, TRIPLE, seed=0)
    assert [(s.prompt_id, s.split) for s in again] == [(s.prompt_id, s.split) for s in v3]


def test_forced_ood_pair():
    specs = build_v3(PAIR, TRIPLE, ood_pair_frac=0.0, ood_pair_frac_in_triple=0.0,
                     forced_ood_pairs=[("elephant", "teddy bear")])
    held = {s.class_set for s in specs if s.split == "test_ood_pair"}
    assert ("elephant", "teddy bear") in held
    assert all({"elephant", "teddy bear"} <= set(c) for c in held)


def test_count_rule_follows_max_count():
    specs = build_v3(PAIR[:6], PAIR[:4], max_count=3)
    assert any(s.split == "test_ood_count" for s in specs)


def test_stratified_subset(v3):
    pool = [s for s in v3 if s.split != "train"]
    sub = stratified_subset(pool, 3000, seed=0)
    assert 2900 <= len(sub) <= 3000
    assert len({s.prompt_id for s in sub}) == len(sub)
    strata = {(s.split, s.n_types, s.max_count) for s in sub}
    all_strata = {(s.split, s.n_types, s.max_count) for s in pool}
    assert strata == all_strata


def test_save_load(tmp_path, v3):
    save_specs(v3[:50], tmp_path / "x.jsonl")
    assert load_specs(tmp_path / "x.jsonl") == v3[:50]
