"""
Detect, score and summarize a run directory (from generate.py or import_legacy_run.py).

    python scripts/score.py --run_dir RUN --detector yolo --threshold 0.5
    python scripts/score.py --run_dir RUN --detector yolo --skip_detect --thresholds_json per_class.json

Detection is resumable and stores every raw detection above --floor, so
re-scoring with other thresholds (--skip_detect) needs no GPU.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.detect import detect_run, score_run
from ioqm.metrics import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--detector", required=True,
                    help="facebook/detr-resnet-50 | facebook/detr-resnet-101 | jozhang97/deta-swin-large | yolo")
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--thresholds_json", default=None, help="per-class {class: threshold}; overrides --threshold")
    ap.add_argument("--skip_detect", action="store_true")
    args = ap.parse_args()

    det_name = args.detector.split("/")[-1]
    if not args.skip_detect:
        from run_eval import get_detector
        n = detect_run(Path(args.run_dir), get_detector(args.detector), det_name, floor=args.floor)
        print(f"detected {n} new images")

    if args.thresholds_json:
        thr, tag = json.loads(Path(args.thresholds_json).read_text()), Path(args.thresholds_json).stem
    else:
        thr, tag = args.threshold, f"t{args.threshold}"
    rows = score_run(Path(args.run_dir), det_name, thr, tag)

    for name, by in (("overall", None),
                     ("by n_types", lambda r: r["n_types"]),
                     ("by max_count", lambda r: r["max_count"]),
                     ("by split", lambda r: r["split"])):
        print(f"\n== {name}")
        for k, ests in summarize(rows, by, metrics=("prompt_exact", "type_exact", "abs_err", "extra_class")).items():
            print(f"  {k}: " + "  ".join(f"{m}={e.mean:.3f}[{e.lo:.3f},{e.hi:.3f}]" for m, e in ests.items())
                  + f"  (prompts={next(iter(ests.values())).n_prompts})")


if __name__ == "__main__":
    main()
