"""
Detection storage and counting.

Raw detections are run ONCE at a low confidence floor and stored with their
scores and boxes. Counting happens afterwards, with a global or per-class
threshold. This means Phase-1 threshold calibration (5-fold CV on human
labels) never re-runs a detector.

    <run_dir>/detections/<detector_name>.jsonl   {"image", "detector", "floor", "dets": [...]}
    <run_dir>/scores/<detector_name>__<tag>.jsonl one ImageScores row per image, plus metadata
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .classes import canonicalize
from .counters import PARSE_ERROR
from .metrics import score_image

Threshold = Union[float, Mapping[str, float]]


def normalize_detections(raw: Iterable[Any]) -> List[dict]:
    """
    Accepts Hugging Face pipeline output ({"label", "score", "box": {xmin, ymin, xmax, ymax}})
    or sahi ObjectPrediction objects (the repo's YOLO/DETA wrappers).
    `label` is the canonical COCO-80 name, or None for labels outside COCO-80.
    """
    out = []
    for r in raw:
        if isinstance(r, Mapping):
            label_raw, score = r["label"], float(r["score"])
            b = r.get("box") or {}
            box = [b.get("xmin"), b.get("ymin"), b.get("xmax"), b.get("ymax")]
        else:  # sahi ObjectPrediction
            label_raw = r.category.name
            score = float(r.score.value)
            bb = r.bbox
            box = [bb.minx, bb.miny, bb.maxx, bb.maxy]
        out.append({"label_raw": label_raw, "label": canonicalize(label_raw), "score": score,
                    "box": [None if v is None else float(v) for v in box]})
    return out


def count_detections(dets: Sequence[Mapping], threshold: Threshold, default: float = 0.5) -> Dict[str, int]:
    """Count detections of each canonical class at or above a global or per-class threshold."""
    counts: Dict[str, int] = {}
    for d in dets:
        name = d["label"]
        if name is None:
            continue
        t = threshold.get(name, default) if isinstance(threshold, Mapping) else threshold
        if d["score"] >= t:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _done_images(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def detect_run(
    run_dir: Path,
    detector: Callable[..., Any],
    detector_name: str,
    floor: float = 0.05,
) -> int:
    """Run `detector(image_path, threshold=floor, classes=...)` on every manifest image
    not yet detected. Resumable. Returns the number of newly processed images."""
    from .generate import images_dir, read_manifest

    run_dir = Path(run_dir)
    img_root = images_dir(run_dir)
    out = run_dir / "detections" / f"{detector_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_images(out)
    n = 0
    with open(out, "a") as f:
        for rec in read_manifest(run_dir):
            if rec["image"] in done:
                continue
            classes = [n for n, _ in rec["objects"]]
            raw = detector(str(img_root / rec["image"]), threshold=floor, classes=classes)
            f.write(json.dumps({"image": rec["image"], "detector": detector_name, "floor": floor,
                                "dets": normalize_detections(raw)}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done.add(rec["image"])
            n += 1
    return n


def load_detections(run_dir: Path, detector_name: str) -> Dict[str, List[dict]]:
    out = {}
    with open(Path(run_dir) / "detections" / f"{detector_name}.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["image"]] = r["dets"]
    return out


def score_run(
    run_dir: Path,
    detector_name: str,
    threshold: Threshold = 0.5,
    tag: str = "t0.5",
    save: bool = True,
) -> List[dict]:
    """Score every manifest image that has detections. Returns rows that
    ioqm.metrics.summarize can consume."""
    from .generate import read_manifest

    run_dir = Path(run_dir)
    dets = load_detections(run_dir, detector_name)
    rows = []
    for rec in read_manifest(run_dir):
        if rec["image"] not in dets:
            continue
        if any(d.get("label_raw") == PARSE_ERROR for d in dets[rec["image"]]):
            continue  # unusable VLM answer: excluded rather than scored as zero counts
        expected = {n: int(c) for n, c in rec["objects"]}
        detected = count_detections(dets[rec["image"]], threshold)
        s = score_image(expected, detected)
        rows.append({
            "image": rec["image"], "prompt_id": rec["prompt_id"], "seed": rec["seed"],
            "model": rec["model"], "split": rec.get("split"), "n_types": len(expected),
            "max_count": max(expected.values()), "expected": expected, "detected": detected,
            "detector": detector_name, "threshold_tag": tag, **s.as_dict(),
        })
    if save:
        p = run_dir / "scores" / f"{detector_name}__{tag}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows
