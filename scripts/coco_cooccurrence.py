"""
COCO-2017 train co-occurrence counts by canonical class name (for PMI features).

    python scripts/coco_cooccurrence.py --out $DATA/coco_cooccurrence.json

Downloads the official annotations zip (~250 MB) to --cache unless it is there.
"""

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ioqm.classes import require_canonical
from ioqm.cooccurrence import build, save

URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/content/coco_annotations_trainval2017.zip")
    args = ap.parse_args()

    zpath = Path(args.cache)
    if not zpath.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        data = json.loads(z.read("annotations/instances_train2017.json"))
    names = {c["id"]: require_canonical(c["name"]) for c in data["categories"]}
    sets = {im["id"]: set() for im in data["images"]}
    for a in data["annotations"]:
        sets[a["image_id"]].add(names[a["category_id"]])
    stats = build(sets.values())
    save(stats, Path(args.out))
    print(f"{stats['n_images']} images, {len(stats['class_counts'])} classes, "
          f"{len(stats['pair_counts'])} co-occurring pairs -> {args.out}")


if __name__ == "__main__":
    main()
