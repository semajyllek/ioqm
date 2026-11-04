"""
Phase 1, step 1: score the counters against the 2023 human labels. No new labelling.

For every 2023 label file (MyDrive/ioqm_data/eval/human_labels/*_human_labels.jsonl)
this builds a small run containing only the labelled images (the images are
referenced, not copied), stores the human counts as the "human" pseudo-counter,
runs each counter on those images, and prints counter-vs-human agreement.
Images labelled twice give an intra-rater check.

    python scripts/human_check.py --data /content/drive/MyDrive/ioqm_data \
        --counters yolov8x mask2former qwen3_vl_8b
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.agreement import agreement, format_agreement, threshold_sweep
from ioqm.counters import COUNTS_ONLY, DEFAULT_THRESHOLDS, get_counter, resolve
from ioqm.detect import detect_run, load_detections, score_run
from ioqm.labels import HUMAN, LEGACY_FILE, dedupe, parse_legacy_label_file, write_human_detections
from ioqm.prompts import make_spec, parse_prompt


def build_run(out: Path, img_folder: Path, labels: dict, version: str, model: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "images_path.txt").write_text(str(img_folder))
    n = 0
    with open(out / "manifest.jsonl", "w") as f:
        for image in sorted(labels):
            if not (img_folder / image).exists():
                continue
            objs = list(parse_prompt(Path(image).stem.replace("_", " ")).items())
            spec = make_spec(objs, dataset=version, split="legacy")
            f.write(json.dumps({"image": image, "prompt_id": spec.prompt_id, "seed": -1,
                                "prompt": Path(image).stem.replace("_", " "),
                                "objects": [list(o) for o in spec.objects], "split": "legacy",
                                "dataset": version, "model": model}) + "\n")
            n += 1
    write_human_detections(out, {k: v for k, v in labels.items() if (img_folder / k).exists()})
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MyDrive/ioqm_data")
    ap.add_argument("--labels_dir", default=None, help="default: <data>/eval/human_labels")
    ap.add_argument("--out_root", default=None, help="default: <data>/runs/human_2023")
    ap.add_argument("--counters", nargs="+", default=["yolov8x", "mask2former"])
    ap.add_argument("--skip_detect", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    labels_dir = Path(args.labels_dir or data / "eval" / "human_labels")
    out_root = Path(args.out_root or data / "runs" / "human_2023")
    counters = [resolve(c) for c in args.counters]

    runs, intra = [], []
    for path in sorted(labels_dir.glob("*_human_labels.jsonl")):
        m = LEGACY_FILE.match(path.name)
        if not m:
            print("skip (unrecognised name):", path.name)
            continue
        folder = m.group("folder")
        good, bad = parse_legacy_label_file(path)
        labels, repeats = dedupe(good)
        intra += repeats
        versions = [m.group("version")] if m.group("version") else ["v1", "v2"]
        if not m.group("version"):
            both = [k for k in labels if all((data / f"generated_images_{v}" / folder / k).exists() for v in ("v1", "v2"))]
            if both:
                print(f"WARNING {path.name}: no v1_/v2_ prefix and {len(both)} labelled file names exist in both "
                      f"generated_images_v1 and _v2; they are attached to v1. Rename the file with the right prefix "
                      f"if you labelled the v2 images.")
        for v in versions:
            img_folder = data / f"generated_images_{v}" / folder
            if not img_folder.is_dir():
                continue
            have = {k: c for k, c in labels.items() if (img_folder / k).exists()}
            if not have:
                continue
            model = folder.replace("_images", "")
            run = out_root / f"{v}_{model}"
            n = build_run(run, img_folder, have, v, model)
            print(f"{path.name}: {len(good)} labels ({len(labels)} images, {len(repeats)} repeats, "
                  f"{len(bad)} unusable) -> {run.name}: {n} images")
            runs.append(run)
            labels = {k: c for k, c in labels.items() if k not in have}

    if not runs:
        sys.exit(f"no label files found in {labels_dir}")

    for c in counters:
        for run in runs:
            if not args.skip_detect:
                detect_run(run, get_counter(c), c)
            score_run(run, c, DEFAULT_THRESHOLDS[c], f"t{DEFAULT_THRESHOLDS[c]}")
    human_rows = {run: score_run(run, HUMAN, 0.5, "t0.5") for run in runs}

    print("\n== Counter vs human (2023 labels; reference = human)")
    all_rows = {c: [] for c in counters}
    all_human = []
    for run in runs:
        print(f"\n-- {run.name}")
        for c in counters:
            rows = score_run(run, c, DEFAULT_THRESHOLDS[c], f"t{DEFAULT_THRESHOLDS[c]}", save=False)
            all_rows[c] += rows
            print(format_agreement(c, agreement(rows, human_rows[run])))
        all_human += human_rows[run]
    print("\n-- all runs pooled")
    for c in counters:
        print(format_agreement(c, agreement(all_rows[c], all_human)))
        if c in COUNTS_ONLY:
            print(f"   ({c}: counts only, so its extra-object rate is not measured)")

    print("\n== Global threshold sweep vs human (pooled; choose by kappa, then confirm with CV)")
    for c in counters:
        if c in COUNTS_ONLY:
            continue
        dets = {}
        for run in runs:
            dets.update(load_detections(run, c))
        print(f"-- {c}")
        for r in threshold_sweep(dets, all_human):
            print(f"   t={r['threshold']:.2f}  kappa={r['kappa']:.3f}  accuracy={r['accuracy']:.3f}  "
                  f"count_agree={r['count_agree']:.3f}  signed_diff={r['signed_diff']:+.3f}  (n={r['n']})")

    if intra:
        same = sum(a == b for _, a, b in intra)
        print(f"\n== Intra-rater: {len(intra)} images labelled twice; identical counts on {same} "
              f"({same / len(intra):.1%})")


if __name__ == "__main__":
    main()
