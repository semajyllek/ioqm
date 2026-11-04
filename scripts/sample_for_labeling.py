"""
Build a human-labelling queue from runs scored by two counters.

    python scripts/sample_for_labeling.py \
        --runs $DATA/runs/bon_pilot/flux2_klein_4b $DATA/runs/v3_eval/z_image_turbo \
        --counters yolov8x mask2former --n 300 --out $DATA/labels/queue_v1.jsonl

The queue is split evenly across runs. Within a run it oversamples images where
the counters disagree but keeps agreement controls (see ioqm/sampling.py).
Re-running with the same arguments gives the same queue. Pass --exclude_queue
to draw a second batch that avoids items already queued.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.counters import DEFAULT_THRESHOLDS, resolve
from ioqm.sampling import DEFAULT_ALLOC, STRATA, interleave, sample_run


def load_scores(run: Path, counter: str):
    p = run / "scores" / f"{counter}__t{DEFAULT_THRESHOLDS[counter]}.jsonl"
    if not p.exists():
        sys.exit(f"{p} missing: run scripts/score.py --run_dir {run} --detector {counter}")
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--counters", nargs=2, default=["yolov8x", "mask2former"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_queue", nargs="*", default=[])
    args = ap.parse_args()

    a_name, b_name = (resolve(c) for c in args.counters)
    exclude = [json.loads(l)["item_id"] for q in args.exclude_queue for l in open(q) if l.strip()]
    per_run = [args.n // len(args.runs) + (i < args.n % len(args.runs)) for i in range(len(args.runs))]

    by_run = {}
    for run, n in zip(args.runs, per_run):
        run_p = Path(run)
        a, b = load_scores(run_p, a_name), load_scores(run_p, b_name)
        by_run[run] = sample_run(str(run_p), a, b, n, a_name, b_name, DEFAULT_ALLOC, args.seed, exclude)
        sizes = collections.Counter(it["stratum"] for it in by_run[run])
        pop = {it["stratum"]: it["stratum_size"] for it in by_run[run]}
        print(f"{run}")
        for s in STRATA:
            print(f"   {s:16s} sampled {sizes.get(s, 0):4d} of {pop.get(s, 0):6d}")

    queue = interleave(by_run, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in queue:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(queue)} items -> {out}")


if __name__ == "__main__":
    main()
