"""
Counter agreement on one or more runs: counter vs counter always, and counter vs
human when human labels exist.

    # after labelling with notebooks/label_colab.ipynb
    python scripts/agreement_report.py --labels $DATA/labels/labels.jsonl \
        --queue $DATA/labels/queue_v1.jsonl --counters yolov8x mask2former qwen3_vl_8b

    # counter vs counter only, on whole runs
    python scripts/agreement_report.py --runs $DATA/runs/bon_pilot/flux2_klein_4b \
        --counters yolov8x mask2former qwen3_vl_8b

With --labels, the human counts are written to each run as the "human"
pseudo-counter, and every estimate is reported twice: on the labelled sample
as drawn, and re-weighted by inverse sampling probability (from --queue) so it
describes the whole run rather than the oversampled disagreements.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.agreement import agreement, format_agreement
from ioqm.counters import COUNTS_ONLY, DEFAULT_THRESHOLDS, resolve
from ioqm.detect import score_run
from ioqm.labels import HUMAN, labels_by_run, read_label_files, write_human_detections
from ioqm.sampling import labeled_weights


def rows_for(run: Path, counter: str):
    if counter == HUMAN:
        return score_run(run, HUMAN, 0.5, "t0.5")
    p = run / "scores" / f"{counter}__t{DEFAULT_THRESHOLDS[counter]}.jsonl"
    if p.exists():
        return [json.loads(l) for l in open(p) if l.strip()]
    return score_run(run, counter, DEFAULT_THRESHOLDS[counter], f"t{DEFAULT_THRESHOLDS[counter]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--labels", nargs="+", default=None,
                    help="one annotator's label files; later files override earlier (e.g. labels.jsonl labels__review.jsonl)")
    ap.add_argument("--queue", default=None, help="queue the labels came from (for weights)")
    ap.add_argument("--counters", nargs="+", default=["yolov8x", "mask2former"])
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    counters = [resolve(c) for c in args.counters]
    labels = {}
    if args.labels:
        recs = list(read_label_files([Path(p) for p in args.labels]).values())
        labels = labels_by_run(recs)
        n_amb = sum(bool(r.get("ambiguous")) for r in recs)
        n_skip = sum(bool(r.get("skipped")) for r in recs)
        print(f"{len(recs)} labelled items: {sum(len(v) for v in labels.values())} usable, "
              f"{n_amb} ambiguous, {n_skip} skipped")
        for run, lab in labels.items():
            write_human_detections(Path(run), lab)
    queue = [json.loads(l) for l in open(args.queue) if l.strip()] if args.queue else []
    runs = list(dict.fromkeys(args.runs + list(labels)))
    if not runs:
        sys.exit("pass --runs and/or --labels")

    for run in runs:
        run_p = Path(run)
        print(f"\n######## {run}")
        rows = {}
        for c in counters:
            try:
                rows[c] = rows_for(run_p, c)
            except FileNotFoundError:
                print(f"   ({c}: no detections in this run, skipped)")
        counters_here = [c for c in counters if c in rows]
        print("\n== counter vs counter (whole run; second is the reference)")
        for a, b in itertools.combinations(counters_here, 2):
            print(format_agreement(f"{a} vs {b}", agreement(rows[a], rows[b], n_boot=args.n_boot)))
        if run in labels:
            human = rows_for(run_p, HUMAN)
            w = labeled_weights(queue, run, [r["image"] for r in human]) if queue else None
            print(f"\n== counter vs human ({len(human)} labelled images)")
            for c in counters_here:
                print(format_agreement(f"{c} (as sampled)", agreement(rows[c], human, n_boot=args.n_boot)))
                if w:
                    print(format_agreement(f"{c} (weighted to run)", agreement(rows[c], human, w, n_boot=args.n_boot)))
                if c in COUNTS_ONLY:
                    print(f"   ({c}: counts only; extra-object rate not measured)")
            if w:
                print("\nweighted: counter_rate vs reference_rate is the counter's bias on the run's headline "
                      "count accuracy; precision is how often a counter's 'correct' verdict is truly correct.")


if __name__ == "__main__":
    main()
