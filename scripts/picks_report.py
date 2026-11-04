"""
Human verdicts on real best-of-k picks: for each (proxy, k), the share of picks
that humans, the gold judge and the proxy itself call correct, on the same images.

    python scripts/picks_report.py --queue $DATA/labels/queue_picks.jsonl \
        --labels $DATA/labels/labels_picks.jsonl

If humans confirm the gold judge's curve, the gap is real; if the gold judge is
also fooled by the picks, the human column shows it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.labels import read_labels, usable
from ioqm.metrics import bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None, help="JSON summary (default: next to the labels)")
    args = ap.parse_args()

    queue = [json.loads(l) for l in open(args.queue) if l.strip()]
    labels = read_labels(Path(args.labels))
    rows = []
    n_unusable = 0
    for it in queue:
        lab = labels.get(it["item_id"])
        if lab is None:
            continue
        if not usable(lab):
            n_unusable += 1
            continue
        human = float(all(int(lab["counts"].get(c, 0)) == n for c, n in it["expected"].items()))
        for role in it["roles"]:
            rows.append({"prompt_id": it["prompt_id"], "proxy": role["proxy"], "k": role["k"],
                         "human": human, "gold": it["gold_prompt_exact"], "proxy_self": role["proxy_prompt_exact"]})
    print(f"{len({r['prompt_id'] for r in rows})} prompts with usable labels; {n_unusable} labelled items "
          f"ambiguous or skipped; {len(queue) - len(labels)} queue items not labelled yet")
    summary = {}
    print(f"\n{'proxy':28s} {'k':>4} | {'human':>21} | {'gold judge':>21} | {'proxy self':>21} | n")
    for proxy in sorted({r["proxy"] for r in rows}):
        for k in sorted({r["k"] for r in rows}):
            rk = [r for r in rows if r["proxy"] == proxy and r["k"] == k]
            if not rk:
                continue
            est = {m: bootstrap(rk, m, n_boot=2000) for m in ("human", "gold", "proxy_self")}
            summary[f"{proxy}|{k}"] = {m: vars(e) for m, e in est.items()}
            f = lambda e: f"{e.mean:.3f} [{e.lo:.3f},{e.hi:.3f}]"
            print(f"{proxy:28s} {k:>4} | {f(est['human']):>21} | {f(est['gold']):>21} | "
                  f"{f(est['proxy_self']):>21} | {len(rk)}")
    out = Path(args.out) if args.out else Path(args.labels).with_name("picks_report.json")
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
