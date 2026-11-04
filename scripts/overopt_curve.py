"""
Best-of-k over-optimization curves: each proxy counter picks, the gold judge scores.

    python scripts/overopt_curve.py --run_dir $DATA/runs/bon_pilot/flux2_klein_4b \
        --proxies yolov8x mask2former ens:yolov8x+mask2former --gold qwen3_vl_8b

Works on a partly finished run (uses the seeds both judges have scored).
Writes <run_dir>/overopt/curves__gold-<gold>.json and a PNG plot next to it.
See ioqm/overopt.py for the estimator.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.counters import resolve
from ioqm.detect import load_scores
from ioqm.overopt import ENSEMBLE_PREFIX, check_independent, curve, ensemble_rows, format_curve, members


def proxy_rows(run: Path, proxy: str):
    if proxy.startswith(ENSEMBLE_PREFIX):
        return ensemble_rows([load_scores(run, resolve(m)) for m in members(proxy)])
    return load_scores(run, resolve(proxy))


def plot(results: dict, path: Path, gold: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (proxy, res) in enumerate(results.items()):
        ks = sorted(int(k) for k in res)
        kl = [res[str(k)]["kl"] for k in ks]
        g = [res[str(k)]["gold"]["mean"] for k in ks]
        lo = [res[str(k)]["gold"]["lo"] for k in ks]
        hi = [res[str(k)]["gold"]["hi"] for k in ks]
        s = [res[str(k)]["proxy_self"]["mean"] for k in ks]
        color = f"C{i}"
        ax.plot(kl, g, "-o", color=color, label=f"{proxy}: gold")
        ax.fill_between(kl, lo, hi, color=color, alpha=0.15)
        ax.plot(kl, s, "--", color=color, label=f"{proxy}: proxy's own score")
    ax.set_xlabel("optimization pressure: KL(best-of-k || base) = log k - (k-1)/k")
    ax.set_ylabel("count accuracy")
    ax.set_title(f"Best-of-k against a counter (gold judge: {gold})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--proxies", nargs="+", default=["yolov8x", "mask2former", "ens:yolov8x+mask2former"])
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--by", default=None, choices=["n_types", "split", "max_count"])
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    run = Path(args.run_dir)
    gold = resolve(args.gold)
    check_independent([p if p.startswith(ENSEMBLE_PREFIX) else resolve(p) for p in args.proxies], gold)
    gold_rows = load_scores(run, gold)

    summary = {}
    for proxy in args.proxies:
        prows = proxy_rows(run, proxy)
        strata = {"all": gold_rows}
        if args.by:
            for v in sorted({r[args.by] for r in gold_rows}, key=str):
                strata[f"{args.by}={v}"] = [r for r in gold_rows if r[args.by] == v]
        for name, grows in strata.items():
            res, info = curve(prows, grows, args.ks, n_boot=args.n_boot)
            print(f"\n== proxy={proxy}  gold={gold}  [{name}]  "
                  f"(prompts={info['prompts']}, images without gold={info['images_without_gold']}, "
                  f"prompt-k pairs skipped for too few seeds={info['skipped_prompt_k']})")
            print(format_curve(res))
            summary.setdefault(name, {})[proxy] = {
                str(k): {m: (vars(v) if hasattr(v, "mean") else v) for m, v in r.items()} for k, r in res.items()}

    out_dir = run / "overopt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"curves__gold-{gold}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}")
    try:
        plot(summary["all"], out_dir / f"curves__gold-{gold}.png", gold)
        print(f"wrote {out.with_suffix('.png')}")
    except ImportError:
        print("(matplotlib not installed: no plot)")


if __name__ == "__main__":
    main()
