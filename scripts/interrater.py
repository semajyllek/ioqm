"""
Agreement between two human annotators on the same queue.

    python scripts/interrater.py --queue $L/queue_v1.jsonl \
        --a $L/labels.jsonl $L/labels__review.jsonl --b $L/labels__labeler_2.jsonl

Each of --a / --b is one annotator's label files, later files overriding earlier
ones. Reports, on items both annotators found usable (not ambiguous, not
skipped): agreement on the prompt-level "all counts exact" verdict and Cohen's
kappa, per-type exact count agreement, mean absolute count difference, and how
often the two agree on flagging an item ambiguous. Estimates are on the queue
as sampled (not re-weighted), which is the usual way to report inter-rater
agreement. --json writes the numbers.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.agreement import agreement, format_agreement
from ioqm.labels import read_label_files, usable


def rows(queue, labels):
    out = {}
    for it in queue:
        lab = labels.get(it["item_id"])
        if lab is None or not usable(lab):
            continue
        det = {c: int(lab["counts"].get(c, 0)) for c in it["expected"]}
        out[it["item_id"]] = {"image": it["item_id"], "expected": it["expected"], "detected": det,
                              "prompt_exact": float(all(det[c] == n for c, n in it["expected"].items()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--a", nargs="+", required=True, help="annotator A's label files")
    ap.add_argument("--b", nargs="+", required=True, help="annotator B's label files")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    queue = [json.loads(l) for l in open(args.queue) if l.strip()]
    la = read_label_files([Path(p) for p in args.a])
    lb = read_label_files([Path(p) for p in args.b])
    both = [it["item_id"] for it in queue if it["item_id"] in la and it["item_id"] in lb]
    if not both:
        sys.exit("no items labelled by both annotators yet")
    amb = lambda r: bool(r.get("ambiguous") or r.get("skipped"))
    flag = [(amb(la[i]), amb(lb[i])) for i in both]
    print(f"{len(both)} items labelled by both; ambiguous/skipped: A {sum(a for a, _ in flag)}, "
          f"B {sum(b for _, b in flag)}, both {sum(a and b for a, b in flag)}")

    ra, rb = rows(queue, la), rows(queue, lb)
    res = agreement(list(ra.values()), list(rb.values()), n_boot=args.n_boot)
    print("\n== A vs B on items both found usable (B is the 'reference' only for the asymmetric stats)")
    print(format_agreement("A vs B", res))
    print("\naccuracy = same all-exact verdict; count_agree = same count per type; abs_diff = mean |A - B| per type")
    if args.json:
        Path(args.json).write_text(json.dumps({"n_both": len(both), "flags": flag, "agreement": res}, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
