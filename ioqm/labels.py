"""
Human count labels.

Human labels are stored as a pseudo-counter called "human": each labelled
instance becomes one detection with score 1.0 in <run_dir>/detections/human.jsonl.
Then ioqm.detect.score_run(run_dir, "human") scores human counts exactly like
any counter, and counter-vs-human comparison is a join on image name.

Two label formats are read:

  New (notebooks/label_colab.ipynb), one JSON object per line:
    {"item_id", "run_dir", "image", "counts": {canonical name: int}, "ambiguous": bool,
     "skipped": bool, "notes", "points", "labeler", "ts"}
  The last line for an item wins, so relabelling simply appends.

  2023 (notebooks/human_ioqm_eval.ipynb), one object per line:
    {"<image file name>": {"<prompt noun>": count, ...}}
  Nouns are the TextBlob plurals of the time ("broccolis", "mouses"). They are
  canonicalized, falling back to the order of the classes in the file name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .classes import canonicalize

HUMAN = "human"


def counts_to_dets(counts: Mapping[str, int]) -> List[dict]:
    return [{"label_raw": n, "label": n, "score": 1.0, "box": [None] * 4}
            for n, c in counts.items() for _ in range(int(c))]


def write_human_detections(run_dir: Path, labels: Mapping[str, Mapping[str, int]]) -> Path:
    """Overwrite <run_dir>/detections/human.jsonl with {image: canonical counts}."""
    out = Path(run_dir) / "detections" / f"{HUMAN}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for image, counts in sorted(labels.items()):
            f.write(json.dumps({"image": image, "detector": HUMAN, "floor": 1.0,
                                "dets": counts_to_dets(counts)}) + "\n")
    return out


# ------------------------------------------------------------------ new format

def read_labels(path: Path) -> Dict[str, dict]:
    """item_id -> latest label record. Missing file -> {}."""
    out: Dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["item_id"]] = r
    return out


def read_label_files(paths: Sequence[Path]) -> Dict[str, dict]:
    """Merge several label files of ONE annotator, later files overriding earlier ones
    (e.g. the original pass, then a re-check under the written rules)."""
    out: Dict[str, dict] = {}
    for p in paths:
        out.update(read_labels(p))
    return out


def labelers_in(path: Path) -> set:
    return {r.get("labeler") for r in read_labels(path).values()}


def needs_review(rec: Mapping) -> bool:
    """Labels made before the written rules that a re-check should revisit."""
    return bool(rec.get("ambiguous") or (rec.get("notes") or "").strip())


def usable(rec: Mapping) -> bool:
    """Labels that count as ground truth: not skipped, not flagged ambiguous."""
    return not rec.get("skipped") and not rec.get("ambiguous") and rec.get("counts") is not None


def labels_by_run(records: Iterable[Mapping]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """run_dir -> image -> counts, usable labels only."""
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in records:
        if usable(r):
            out.setdefault(r["run_dir"], {})[r["image"]] = {k: int(v) for k, v in r["counts"].items()}
    return out


# ------------------------------------------------------------------ 2023 format

LEGACY_FILE = re.compile(r"^(?:(?P<version>v[12])_)?(?P<folder>.+_images)_human_labels\.jsonl$")


def parse_legacy_label_file(path: Path) -> Tuple[List[Tuple[str, Dict[str, int]]], List[Tuple[str, str]]]:
    """Returns ([(image name, canonical counts)], [(image name, problem)]) in file order.
    Repeated images are kept; callers decide (they are the intra-rater pairs)."""
    from .prompts import parse_prompt
    good, bad = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad.append(("?", "bad json"))
                continue
            for image, raw in obj.items():
                try:
                    expected_order = list(parse_prompt(Path(image).stem.replace("_", " ")))
                except ValueError as e:
                    bad.append((image, f"unparseable file name: {e}"))
                    continue
                counts: Dict[str, int] = {}
                problem = None
                for i, (noun, c) in enumerate(raw.items()):
                    name = canonicalize(noun)
                    if name is None and i < len(expected_order):
                        name = expected_order[i]
                    if name is None:
                        problem = f"unknown noun {noun!r}"
                        break
                    if c is None:
                        problem = f"missing count for {noun!r}"
                        break
                    counts[name] = int(round(float(c)))
                if problem:
                    bad.append((image, problem))
                else:
                    good.append((image, counts))
    return good, bad


def dedupe(pairs: Iterable[Tuple[str, Dict[str, int]]]) -> Tuple[Dict[str, Dict[str, int]], List[Tuple[str, Dict[str, int], Dict[str, int]]]]:
    """Keep the last label per image. Also return (image, first, last) for images labelled
    more than once: the intra-rater pairs."""
    first: Dict[str, Dict[str, int]] = {}
    last: Dict[str, Dict[str, int]] = {}
    n: Dict[str, int] = {}
    for image, counts in pairs:
        first.setdefault(image, counts)
        last[image] = counts
        n[image] = n.get(image, 0) + 1
    repeats = [(im, first[im], last[im]) for im, k in n.items() if k > 1]
    return last, repeats
