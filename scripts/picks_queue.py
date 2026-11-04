"""
Queue of real best-of-k picks for human verification of the over-optimization curve.

    python scripts/picks_queue.py --run_dir $DATA/runs/bon_pilot/flux2_klein_4b \
        --proxies yolov8x mask2former ens:yolov8x+mask2former --gold qwen3_vl_8b \
        --ks 1 64 --n_prompts 50 --out $DATA/labels/queue_picks.jsonl

Label it with notebooks/label_colab.ipynb (set QUEUE to this file), then run
scripts/picks_report.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.counters import resolve
from ioqm.detect import load_scores
from ioqm.overopt import ENSEMBLE_PREFIX, check_independent, picks_queue
from overopt_curve import proxy_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--proxies", nargs="+", default=["yolov8x", "mask2former", "ens:yolov8x+mask2former"])
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--n_prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    gold = resolve(args.gold)
    check_independent([p if p.startswith(ENSEMBLE_PREFIX) else resolve(p) for p in args.proxies], gold)
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite a queue that may already be partly labelled")
    q = picks_queue(str(run), {p: proxy_rows(run, p) for p in args.proxies}, load_scores(run, gold),
                    ks=args.ks, n_prompts=args.n_prompts, seed=args.seed)
    n_prompts = len({it["prompt_id"] for it in q})
    if n_prompts < args.n_prompts:
        print(f"only {n_prompts} prompts have all {max(args.ks)} seeds scored by every judge "
              f"(need {args.n_prompts}); not writing a queue yet. Finish generation and scoring first.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in q:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(q)} images ({len({it['prompt_id'] for it in q})} prompts) -> {out}")


if __name__ == "__main__":
    main()
