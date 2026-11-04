"""
Mechanism analyses (no GPU): are per-type errors independent, do types trade
instances, and what predicts failure. See ioqm/analysis.py.

    python scripts/mechanism.py --run_dir $DATA/runs/bon_pilot/flux2_klein_4b \
        --judges qwen3_vl_8b yolov8x mask2former --cooccurrence $DATA/coco_cooccurrence.json

Dependence needs many seeds per prompt (default >= 8), so use the best-of-N
run. The regression works on any run. Results are reported per judge: a
finding is only credible if it holds under every judge (and, later, humans).
Writes <run_dir>/mechanism/summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.analysis import dependence, failure_regression
from ioqm.counters import resolve
from ioqm.detect import load_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--judges", nargs="+", default=["yolov8x", "mask2former"])
    ap.add_argument("--cooccurrence", default=None)
    ap.add_argument("--min_seeds", type=int, default=8)
    ap.add_argument("--n_perm", type=int, default=500)
    args = ap.parse_args()

    run = Path(args.run_dir)
    pmi = None
    if args.cooccurrence and Path(args.cooccurrence).exists():
        from ioqm.cooccurrence import load, pmi_fn
        pmi = pmi_fn(load(Path(args.cooccurrence)))
    elif args.cooccurrence:
        print(f"(no co-occurrence file at {args.cooccurrence}; regression runs without PMI)")

    summary = {}
    for judge in [resolve(j) for j in args.judges]:
        rows = load_scores(run, judge)
        print(f"\n######## judge = {judge}  ({len(rows)} images)")
        summary[judge] = {}
        for nt in (2, 3):
            sub = [r for r in rows if r["n_types"] == nt]
            try:
                d = dependence(sub, min_seeds=args.min_seeds, n_perm=args.n_perm)
            except ValueError as e:
                print(f"  {nt} types: {e}")
                continue
            summary[judge][f"dependence_{nt}types"] = d
            print(f"  {nt} types, {d['n_prompts']} prompts:  P(all exact) = {d['joint']:.3f}  vs product of "
                  f"per-type rates = {d['product']:.3f}  (ratio {d['joint_over_product']:.2f}, "
                  f"p = {d['p_independence']:.3g})")
            print(f"     signed-error correlation between types = {d['trade_corr']:+.3f} "
                  f"(null {d['null_mean_trade']:+.3f}, p = {d['p_trade']:.3g}; negative = trading)")
        try:
            res, names = failure_regression(rows, pmi)
            tab = {n: {"coef": float(c), "lo": float(lo), "hi": float(hi), "p": float(p)}
                   for n, c, (lo, hi), p in zip(names, res.params, res.conf_int(), res.pvalues)}
            summary[judge]["regression"] = tab
            print("  failure regression (multi-type prompts; log-odds of exact per SD; SEs clustered by prompt)")
            if res.dropped_features:
                print(f"     dropped (collinear with the rest): {', '.join(res.dropped_features)}")
                summary[judge]["regression_dropped"] = res.dropped_features
            for n, t in tab.items():
                print(f"     {n:16s} {t['coef']:+.3f}  [{t['lo']:+.3f}, {t['hi']:+.3f}]  p={t['p']:.3g}")
        except Exception as e:   # e.g. statsmodels missing or a degenerate subset
            print(f"  regression failed: {type(e).__name__}: {e}")

    out = run / "mechanism" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
