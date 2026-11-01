"""
Canonical COCO-80 class table for IOQM.

The class's canonical name is the COCO-2017 label string. The DETR/DETA Hugging
Face configs and the Ultralytics YOLO config all use these same strings. Each
class also carries:
  - prompt_singular / prompt_plural: what v3 prompts say. This is sometimes
    different from the COCO label, to remove ambiguity: COCO "mouse" is a
    computer mouse, not the animal.
  - aliases: other surface forms seen in legacy (2023) prompts, human labels or
    detectors, e.g. "people", "mouses", "sheeps", "broccolis", "tvs", "ski".
  - unit_ambiguous: the countable unit is unclear ("3 skis": 3 skis, or 3
    pairs?). COCO annotates a pair of skis as one box. These classes are left
    out of the default v3 class pool.

All lookup goes through `canonicalize`. Nothing here uses prefix matching: the
2023 prefix matcher matched "car" to "carrot" and never matched "people" to
"person".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class CocoClass:
    coco_id: int                 # contiguous 0..79 index (Ultralytics order)
    name: str                    # canonical COCO-2017 label
    prompt_singular: str
    prompt_plural: str
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    unit_ambiguous: bool = False
    notes: str = ""


def _c(i, name, plural=None, singular=None, aliases=(), unit_ambiguous=False, notes=""):
    singular = singular or name
    return CocoClass(
        coco_id=i,
        name=name,
        prompt_singular=singular,
        prompt_plural=plural or singular + "s",
        aliases=tuple(aliases),
        unit_ambiguous=unit_ambiguous,
        notes=notes,
    )


# Plurals are written out by hand, not generated with TextBlob.
COCO80: Tuple[CocoClass, ...] = (
    _c(0, "person", "people", aliases=("persons", "man", "men", "woman", "women")),
    _c(1, "bicycle", aliases=("bike", "bikes")),
    _c(2, "car"),
    _c(3, "motorcycle", aliases=("motorbike", "motorbikes")),
    _c(4, "airplane", aliases=("aeroplane", "aeroplanes", "plane", "planes")),
    _c(5, "bus", "buses", aliases=("busses",)),
    _c(6, "train"),
    _c(7, "truck"),
    _c(8, "boat"),
    _c(9, "traffic light"),
    _c(10, "fire hydrant"),
    _c(11, "stop sign"),
    _c(12, "parking meter"),
    _c(13, "bench", "benches"),
    _c(14, "bird"),
    _c(15, "cat"),
    _c(16, "dog"),
    _c(17, "horse"),
    _c(18, "sheep", "sheep", aliases=("sheeps",)),
    _c(19, "cow"),
    _c(20, "elephant"),
    _c(21, "bear"),
    _c(22, "zebra"),
    _c(23, "giraffe"),
    _c(24, "backpack"),
    _c(25, "umbrella"),
    _c(26, "handbag"),
    _c(27, "tie", aliases=("ty", "necktie", "neckties")),
    _c(28, "suitcase"),
    _c(29, "frisbee"),
    _c(30, "skis", "pairs of skis", singular="pair of skis", aliases=("ski",),
       unit_ambiguous=True, notes="COCO boxes a pair of skis as one instance"),
    _c(31, "snowboard"),
    _c(32, "sports ball", aliases=("ball", "balls")),
    _c(33, "kite"),
    _c(34, "baseball bat"),
    _c(35, "baseball glove"),
    _c(36, "skateboard"),
    _c(37, "surfboard"),
    _c(38, "tennis racket"),
    _c(39, "bottle"),
    _c(40, "wine glass", "wine glasses"),
    _c(41, "cup"),
    _c(42, "fork"),
    _c(43, "knife", "knives"),
    _c(44, "spoon"),
    _c(45, "bowl"),
    _c(46, "banana"),
    _c(47, "apple"),
    _c(48, "sandwich", "sandwiches"),
    _c(49, "orange"),
    _c(50, "broccoli", "heads of broccoli", singular="head of broccoli",
       aliases=("broccolis",), unit_ambiguous=True,
       notes="mass noun; florets vs heads is ambiguous"),
    _c(51, "carrot"),
    _c(52, "hot dog"),
    _c(53, "pizza"),
    _c(54, "donut", aliases=("doughnut", "doughnuts")),
    _c(55, "cake"),
    _c(56, "chair"),
    _c(57, "couch", "couches", aliases=("sofa", "sofas")),
    _c(58, "potted plant", aliases=("pottedplant",)),
    _c(59, "bed"),
    _c(60, "dining table", aliases=("diningtable",)),
    _c(61, "toilet"),
    _c(62, "tv", "TVs", singular="TV", aliases=("tvs", "tvmonitor", "television", "televisions")),
    _c(63, "laptop"),
    _c(64, "mouse", "computer mice", singular="computer mouse",
       aliases=("mouses", "mice"), notes="COCO mouse = computer mouse"),
    _c(65, "remote", "remote controls", singular="remote control", aliases=("remotes",)),
    _c(66, "keyboard"),
    _c(67, "cell phone", aliases=("cellphone", "cellphones", "phone", "phones")),
    _c(68, "microwave"),
    _c(69, "oven"),
    _c(70, "toaster"),
    _c(71, "sink"),
    _c(72, "refrigerator", aliases=("fridge", "fridges")),
    _c(73, "book"),
    _c(74, "clock"),
    _c(75, "vase"),
    _c(76, "scissors", "pairs of scissors", singular="pair of scissors",
       unit_ambiguous=True, notes="plural-only noun"),
    _c(77, "teddy bear"),
    _c(78, "hair drier", "hair dryers", singular="hair dryer",
       aliases=("hair driers", "hairdryer", "hair dryer")),
    _c(79, "toothbrush", "toothbrushes"),
)

BY_NAME: Dict[str, CocoClass] = {c.name: c for c in COCO80}
BY_ID: Dict[int, CocoClass] = {c.coco_id: c for c in COCO80}

# Labels that appear in DETR's 91-slot id2label but have no COCO-2017
# annotations. They are not COCO-80 classes and are dropped, not counted.
NON_COCO80_DETECTOR_LABELS = frozenset({
    "N/A", "blender", "desk", "door", "eye glasses", "hat", "mirror",
    "plate", "shoe", "street sign", "window",
})


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _build_surface_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for c in COCO80:
        forms = {c.name, c.prompt_singular, c.prompt_plural, *c.aliases}
        forms |= {c.name + "s"} if not c.name.endswith("s") else set()
        for f in forms:
            key = _norm(f)
            if key in m and m[key] != c.name:
                raise ValueError(f"surface form {f!r} maps to both {m[key]} and {c.name}")
            m[key] = c.name
    return m


SURFACE_TO_NAME: Dict[str, str] = _build_surface_map()
# Longest first, so the longest multi-word surface form wins (e.g. "hot dog" before "dog").
SURFACE_FORMS_BY_LENGTH = sorted(SURFACE_TO_NAME, key=len, reverse=True)


def canonicalize(label: str) -> Optional[str]:
    """Map any surface form (detector label, prompt noun, legacy string) to the
    canonical COCO name. Returns None for non-COCO-80 detector labels and for
    unknown strings."""
    if label in NON_COCO80_DETECTOR_LABELS:
        return None
    return SURFACE_TO_NAME.get(_norm(label))


def require_canonical(label: str) -> str:
    name = canonicalize(label)
    if name is None:
        raise KeyError(f"unknown COCO-80 class surface form: {label!r}")
    return name


def prompt_noun(name: str, count: int) -> str:
    c = BY_NAME[name]
    return c.prompt_singular if count == 1 else c.prompt_plural


def canonical_counts(counts: Dict[str, int], strict: bool = True) -> Dict[str, int]:
    """Re-key a {surface_form: count} dict by canonical name, merging duplicates."""
    out: Dict[str, int] = {}
    for k, v in counts.items():
        name = require_canonical(k) if strict else canonicalize(k)
        if name is None:
            continue
        out[name] = out.get(name, 0) + int(v)
    return out


def names(classes: Iterable[str]) -> Tuple[str, ...]:
    return tuple(require_canonical(c) for c in classes)
