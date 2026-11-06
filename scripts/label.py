"""
Run the labeller on this computer, reading images from Google Drive for desktop.

    python3 -m venv .venv && .venv/bin/pip install gradio     # once, isolated from other projects
    .venv/bin/python scripts/label.py                    # round 1: queue_v1 -> labels.jsonl
    .venv/bin/python scripts/label.py --round picks      # round 2: queue_picks -> labels_picks.jsonl

    # second annotator: own files (labels__labeler_2.jsonl, labels_picks__labeler_2.jsonl)
    .venv/bin/python scripts/label.py --labeler labeler_2
    .venv/bin/python scripts/label.py --labeler labeler_2 --round picks

    # re-check, under docs/labeling_rubric.md, the items you flagged ambiguous or left a
    # note on before the rules existed (-> labels__review.jsonl; overrides the originals)
    .venv/bin/python scripts/label.py --review
    .venv/bin/python scripts/label.py --review --round picks

Opens in your browser. Queue items carry Colab paths (/content/drive/MyDrive/...);
they are mapped to the local Drive folder, so only the queued images are
downloaded (with Drive set to "Stream files"). Labels are written to the same
Drive folder, in the same format as the Colab labeller, so either can continue
the other's work. Don't label the same queue in Colab and here at the same time.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.labeler import build_app, launch, load_queue
from ioqm.labels import labelers_in, needs_review, read_labels

COLAB_ROOT = "/content/drive/MyDrive"
ROUNDS = {"v1": ("queue_v1.jsonl", "labels.jsonl"), "picks": ("queue_picks.jsonl", "labels_picks.jsonl")}
FIRST = "labeler_1"


def labels_file(label_dir: Path, rnd: str, labeler: str, review: bool) -> Path:
    """labeler_1 keeps the original names; anyone else, and re-checks, get their own file."""
    base = Path(ROUNDS[rnd][1])
    if review:
        return label_dir / f"{base.stem}__review{base.suffix}"
    if labeler == FIRST:
        return label_dir / base.name
    return label_dir / f"{base.stem}__{labeler}{base.suffix}"


def find_drive() -> str:
    hits = sorted(glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive")))
    if not hits:
        sys.exit("No Google Drive for desktop folder found under ~/Library/CloudStorage; pass --drive_root")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=sorted(ROUNDS), default="v1")
    ap.add_argument("--drive_root", default=None, help="local folder that corresponds to Colab's MyDrive")
    ap.add_argument("--queue", default=None, help="override the round's queue file")
    ap.add_argument("--labels", default=None, help="override the round's labels file")
    ap.add_argument("--labeler", default=FIRST, help="annotator id (not a name), e.g. labeler_2")
    ap.add_argument("--review", action="store_true",
                    help="re-check this labeller's ambiguous or noted items under the written rules")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    drive = args.drive_root or find_drive()
    label_dir = Path(drive) / "ioqm_data" / "labels"
    queue = Path(args.queue or label_dir / ROUNDS[args.round][0])
    labels = Path(args.labels or labels_file(label_dir, args.round, args.labeler, args.review))
    if not queue.exists():
        sys.exit(f"queue not found: {queue}\n(build it in Colab first: main notebook section 5b, or overopt step H)")
    others = labelers_in(labels) - {args.labeler}
    if others:
        sys.exit(f"{labels} already holds labels from {sorted(others)}; pick another --labels file "
                 f"so two annotators never share one")
    items = None
    if args.review:
        if args.labeler != FIRST:
            sys.exit("--review re-checks labeler_1's pre-rules labels")
        base = read_labels(label_dir / ROUNDS[args.round][1])
        items = [it for it in load_queue(queue) if it["item_id"] in base and needs_review(base[it["item_id"]])]
        print(f"{len(items)} items to re-check (flagged ambiguous or with a note)")
    print(f"queue  {queue}\nlabels {labels}")
    app = build_app(queue, labels, args.labeler, path_map=(COLAB_ROOT, drive), queue=items)
    launch(app, inbrowser=True, server_port=args.port)


if __name__ == "__main__":
    main()
