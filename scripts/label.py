"""
Run the labeller on this computer, reading images from Google Drive for desktop.

    python3 -m venv .venv && .venv/bin/pip install gradio     # once, isolated from other projects
    .venv/bin/python scripts/label.py                    # round 1: queue_v1 -> labels.jsonl
    .venv/bin/python scripts/label.py --round picks      # round 2: queue_picks -> labels_picks.jsonl

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

from ioqm.labeler import build_app, launch

COLAB_ROOT = "/content/drive/MyDrive"
ROUNDS = {"v1": ("queue_v1.jsonl", "labels.jsonl"), "picks": ("queue_picks.jsonl", "labels_picks.jsonl")}


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
    ap.add_argument("--labeler", default="labeler_1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    drive = args.drive_root or find_drive()
    label_dir = Path(drive) / "ioqm_data" / "labels"
    queue = Path(args.queue or label_dir / ROUNDS[args.round][0])
    labels = Path(args.labels or label_dir / ROUNDS[args.round][1])
    if not queue.exists():
        sys.exit(f"queue not found: {queue}\n(build it in Colab first: main notebook section 5b, or overopt step H)")
    print(f"queue  {queue}\nlabels {labels}")
    app = build_app(queue, labels, args.labeler, path_map=(COLAB_ROOT, drive))
    launch(app, inbrowser=True, server_port=args.port)


if __name__ == "__main__":
    main()
