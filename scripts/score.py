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

from ioqm.counters import COUNTS_ONLY, DEFAULT_THRESHOLDS, get_counter, resolve
from ioqm.detect import detect_run, score_run
from ioqm.metrics import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--detector", required=True,
                    help="counter key from ioqm.counters: yolov8x | mask2former | detr_r50 | detr_r101 | deta_swin_large "
                         "(2023 ids such as 'yolo' are accepted too)")
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=None, help="default: the counter's DEFAULT_THRESHOLDS value")
    ap.add_argument("--thresholds_json", default=None, help="per-class {class: threshold}; overrides --threshold")
    ap.add_argument("--skip_detect", action="store_true")
    ap.add_argument("--from_queue", default=None,
                    help="labelling queue: detect only this run's queued images (e.g. a slow VLM counter)")
    args = ap.parse_args()

    det_name = resolve(args.detector)
    if args.threshold is None:
        args.threshold = DEFAULT_THRESHOLDS[det_name]
    if not args.skip_detect:
        only = None
        if args.from_queue:
            run_key = str(Path(args.run_dir))
            only = [json.loads(l)["image"] for l in open(args.from_queue)
                    if l.strip() and str(Path(json.loads(l)["run_dir"])) == run_key]
            print(f"{len(only)} queued images from this run")
        n = detect_run(Path(args.run_dir), get_counter(det_name), det_name, floor=args.floor, only=only)
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
        metrics = ("prompt_exact", "type_exact", "abs_err") + (() if det_name in COUNTS_ONLY else ("extra_class",))
        for k, ests in summarize(rows, by, metrics=metrics).items():
            print(f"  {k}: " + "  ".join(f"{m}={e.mean:.3f}[{e.lo:.3f},{e.hi:.3f}]" for m, e in ests.items())
                  + f"  (prompts={next(iter(ests.values())).n_prompts})")


if __name__ == "__main__":
    main()
