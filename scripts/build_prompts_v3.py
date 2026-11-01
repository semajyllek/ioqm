"""
Build the IOQM v3 prompt set (nested exhaustive) and its stratified evaluation
subset, and convert the legacy v1/v2 prompt files to the same JSONL format.

    python scripts/build_prompts_v3.py --out_dir prompts/v3

The class lists below are PROVISIONAL. Per docs/plan.md, the final 24 / 8
classes are chosen by stratifying on (a) per-class counter accuracy from
Phase 1 and (b) COCO co-occurrence frequency (label_combos.npy). Override them
with --pair_classes / --triple_classes, or edit the lists and rebuild. Prompt
IDs are content hashes, so they stay stable across rebuilds.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.classes import BY_NAME
from ioqm.prompts import build_v3, legacy_specs, save_specs, stratified_subset

# Excludes the unit-ambiguous classes (skis, scissors, broccoli). Mixes large and
# small objects, animate and inanimate.
PROVISIONAL_PAIR_CLASSES = [
    "person", "bicycle", "car", "bus", "boat", "dog", "cat", "horse", "sheep", "elephant",
    "bird", "umbrella", "backpack", "suitcase", "bottle", "wine glass", "cup", "bowl",
    "banana", "apple", "chair", "potted plant", "teddy bear", "clock",
]
# Spans common COCO co-occurrences (person+dog, cup+chair) and rare ones (elephant+teddy bear, apple+umbrella).
PROVISIONAL_TRIPLE_CLASSES = ["person", "dog", "car", "cup", "apple", "elephant", "umbrella", "teddy bear"]

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO / "prompts" / "v3"))
    ap.add_argument("--pair_classes", nargs="*", default=PROVISIONAL_PAIR_CLASSES)
    ap.add_argument("--triple_classes", nargs="*", default=PROVISIONAL_TRIPLE_CLASSES)
    ap.add_argument("--max_count", type=int, default=5)
    ap.add_argument("--eval_subset_size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot_size", type=int, default=480,
                    help="stratified sub-sample of the eval subset for best-of-N pilots (x16 seeds)")
    args = ap.parse_args()

    for c in args.pair_classes:
        if BY_NAME[c].unit_ambiguous:
            print(f"warning: {c!r} has an ambiguous counting unit", file=sys.stderr)

    out = Path(args.out_dir)
    specs = build_v3(args.pair_classes, args.triple_classes, args.max_count, seed=args.seed)
    for split in ("train", "test_id", "test_single", "test_ood_pair", "test_ood_triple", "test_ood_count"):
        assert any(s.split == split for s in specs), f"split {split!r} is empty; check the build parameters"
    singles = [s for s in specs if s.split == "test_single"]          # always evaluated in full
    eval_pool = [s for s in specs if s.split not in ("train", "test_single")]
    subset = sorted(singles + stratified_subset(eval_pool, args.eval_subset_size - len(singles), seed=args.seed),
                    key=lambda s: s.prompt_id)

    save_specs(specs, out / "all.jsonl")
    save_specs([s for s in specs if s.split == "train"], out / "train.jsonl")
    save_specs(subset, out / "eval_subset.jsonl")
    pilot = stratified_subset(subset, args.pilot_size, key=lambda s: (s.split, s.n_types), seed=args.seed)
    save_specs(pilot, out / "pilot.jsonl")

    summary = {
        "pair_classes": args.pair_classes, "triple_classes": args.triple_classes,
        "max_count": args.max_count, "seed": args.seed,
        "n_total": len(specs),
        "by_split": dict(Counter(s.split for s in specs)),
        "by_n_types": dict(Counter(s.n_types for s in specs)),
        "eval_subset": {"n": len(subset),
                        "by_split": dict(Counter(s.split for s in subset)),
                        "by_n_types": dict(Counter(s.n_types for s in subset)),
                        "by_max_count": dict(Counter(s.max_count for s in subset))},
        "pilot": {"n": len(pilot), "by_split": dict(Counter(s.split for s in pilot)),
                  "by_n_types": dict(Counter(s.n_types for s in pilot))},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # legacy prompt files -> JSONL specs (canonical classes, fixed parsing)
    for name, fname in (("v1", "prompts_v1.txt"), ("v2", "prompts_v2.txt")):
        p = REPO / "prompts" / fname
        if p.exists():
            ls = legacy_specs(p.read_text().splitlines(), dataset=name)
            save_specs(ls, REPO / "prompts" / f"legacy_{name}.jsonl")
            print(f"legacy {name}: {len(ls)} prompts -> prompts/legacy_{name}.jsonl")


if __name__ == "__main__":
    main()
