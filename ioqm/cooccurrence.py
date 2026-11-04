"""
COCO-2017 train co-occurrence statistics, keyed by canonical class name.

Built once by scripts/coco_cooccurrence.py from the official annotations and
stored as JSON: {"n_images", "class_counts": {name: n}, "pair_counts": {"a|b": n}}
(a < b alphabetically; counts are numbers of images containing both).

PMI(a, b) = log( n_images * n(a,b) / (n(a) * n(b)) ), with add-one smoothing
on n(a,b) so never-co-occurring pairs get a finite, very negative value.
For a triple we use the minimum pairwise PMI (the rarest pairing), which is a
choice, not a property of the data: say so when reporting.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence, Set


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def build(image_class_sets: Iterable[Set[str]]) -> dict:
    n = 0
    cc: Dict[str, int] = {}
    pc: Dict[str, int] = {}
    for s in image_class_sets:
        n += 1
        for c in s:
            cc[c] = cc.get(c, 0) + 1
        for a, b in itertools.combinations(sorted(s), 2):
            k = pair_key(a, b)
            pc[k] = pc.get(k, 0) + 1
    return {"n_images": n, "class_counts": cc, "pair_counts": pc}


def save(stats: Mapping, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats))


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def pmi_fn(stats: Mapping) -> Callable[[Sequence[str]], float]:
    n = stats["n_images"]
    cc, pc = stats["class_counts"], stats["pair_counts"]

    def pmi_pair(a: str, b: str) -> float:
        return math.log(n * (pc.get(pair_key(a, b), 0) + 1) / ((cc.get(a, 0) + 1) * (cc.get(b, 0) + 1)))

    def min_pmi(classes: Sequence[str]) -> float:
        return min(pmi_pair(a, b) for a, b in itertools.combinations(classes, 2))
    return min_pmi
