"""
Best-of-N count accuracy for a run generated with seeds 0..N-1, for every
(verifier, evaluator) pair of counters that have scores in the run.

Prerequisites (see notebooks/phase0_colab.ipynb, Best-of-N section):
    python scripts/generate.py --model z_image_turbo --prompts prompts/v3/pilot.jsonl \
        --seeds $(seq 0 15) --out_dir RUN
    python scripts/score.py --run_dir RUN --detector yolov8x
    python scripts/score.py --run_dir RUN --detector mask2former

Then:
    python scripts/best_of_n.py --run_dir RUN --counters yolov8x mask2former

Pairs where verifier == evaluator are printed too, but labelled SELF: they show
how optimistic self-selection is. Only verifier != evaluator rows are results.
Writes RUN/best_of_n/summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.best_of_n import curve, format_curve
from ioqm.counters import DEFAULT_THRESHOLDS, resolve


def load_scores(run_dir: Path, counter: str, tag: str):
    p = run_dir / "scores" / f"{counter}__{tag}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/score.py --run_dir {run_dir} --detector {counter}")
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--counters", nargs="+", required=True)
    ap.add_argument("--tags", nargs="*", default=None,
                    help="score-file tag per counter (default t<DEFAULT_THRESHOLD>, as score.py writes)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--by", default=None, choices=[None, "n_types", "split", "max_count"],
                    help="also report curves per stratum")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    run = Path(args.run_dir)
    counters = [resolve(c) for c in args.counters]
    tags = args.tags or [f"t{DEFAULT_THRESHOLDS[c]}" for c in counters]
    scores = {c: load_scores(run, c, t) for c, t in zip(counters, tags)}

    summary = {}
    for v in counters:
        for e in counters:
            label = f"verifier={v}  evaluator={e}" + ("   [SELF: optimistic, not a result]" if v == e else "")
            strata = {"all": (scores[v], scores[e])}
            if args.by:
                keys = sorted({r[args.by] for r in scores[e]}, key=str)
                strata.update({f"{args.by}={k}": ([r for r in scores[v] if r[args.by] == k],
                                                  [r for r in scores[e] if r[args.by] == k]) for k in keys})
            for name, (vr, er) in strata.items():
                res = curve(vr, er, args.ks, n_boot=args.n_boot)
                print(f"\n== {label}   [{name}]")
                print(format_curve(res))
                summary[f"{v}|{e}|{name}"] = {
                    str(k): {m: (vars(x) if hasattr(x, "mean") else x) for m, x in r.items()} for k, r in res.items()}

    out = run / "best_of_n" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
