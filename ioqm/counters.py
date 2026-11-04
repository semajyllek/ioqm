"""
Counters (detectors) used as evaluators or best-of-N verifiers.

Every counter is a callable `counter(image_path, threshold, classes=None) -> list[dict]`,
each dict being {"label", "score", "box": {xmin, ymin, xmax, ymax}}. That is the
Hugging Face pipeline format, which ioqm.detect.normalize_detections accepts.
`classes` (canonical names of the requested objects) is ignored by detectors and
used by VLM counters, which are asked about those classes only.
Labels are canonicalized downstream (ioqm.classes), so the older names
Mask2Former uses ("aeroplane", "tvmonitor", ...) map correctly.

DEFAULT_THRESHOLDS are pre-calibration starting points. Phase 1 replaces them
with per-class thresholds fitted to human labels. Detections are stored at a
low floor, so changing thresholds never requires re-running a counter.

Heavy imports are lazy, so `import ioqm.counters` works without torch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

Counter = Callable[..., List[dict]]

COUNTERS = {
    # key: (description, default threshold, source of the default)
    "yolov8x": ("Ultralytics YOLOv8x, COCO-80", 0.5, "2023 IOQM default"),
    "mask2former": ("Mask2Former Swin-L COCO instance (HF facebook/mask2former-swin-large-coco-instance)", 0.9,
                    "GenEval's counting threshold. GenEval's scorer is also Mask2Former, but its exact "
                    "MMDetection checkpoint may differ from this HF one"),
    "detr_r50": ("DETR ResNet-50 (HF facebook/detr-resnet-50)", 0.5, "HF object-detection pipeline default"),
    "detr_r101": ("DETR ResNet-101 (HF facebook/detr-resnet-101)", 0.5, "HF object-detection pipeline default"),
    "deta_swin_large": ("DETA Swin-L (HF jozhang97/deta-swin-large)", 0.2, "2023 IOQM default"),
    # VLM counter: an independent third judge with different training data and failure
    # modes from the two COCO detectors. It returns counts, not boxes: each counted
    # instance becomes one detection with score 1.0 and no box, so any threshold <= 1
    # gives the same counts. It is asked only about the requested classes, so its
    # extra_class score is always 0 and must not be reported.
    "qwen3_vl_8b": ("Qwen3-VL-8B-Instruct asked for per-class counts (HF Qwen/Qwen3-VL-8B-Instruct)", 0.5,
                    "threshold unused: every VLM count has score 1.0"),
}
COUNTS_ONLY = {"qwen3_vl_8b"}   # counters whose extra_class is meaningless
PARSE_ERROR = "__counter_parse_error__"   # label emitted when a VLM answer cannot be parsed
DEFAULT_THRESHOLDS: Dict[str, float] = {k: v[1] for k, v in COUNTERS.items()}

# 2023 detector ids -> counter keys
LEGACY_ALIASES = {
    "yolo": "yolov8x",
    "facebook/detr-resnet-50": "detr_r50",
    "facebook/detr-resnet-101": "detr_r101",
    "jozhang97/deta-swin-large": "deta_swin_large",
}


def resolve(name: str) -> str:
    key = LEGACY_ALIASES.get(name, name)
    if key not in COUNTERS:
        raise KeyError(f"unknown counter {name!r}; choose from {sorted(COUNTERS)}")
    return key


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _box(x1, y1, x2, y2) -> dict:
    return {"xmin": float(x1), "ymin": float(y1), "xmax": float(x2), "ymax": float(y2)}


@lru_cache(maxsize=None)
def get_counter(name: str) -> Counter:
    key = resolve(name)

    if key == "yolov8x":
        from ultralytics import YOLO
        model = YOLO("yolov8x.pt")

        def yolo(image_path: str, threshold: float, classes=None) -> List[dict]:
            r = model(image_path, imgsz=640, conf=threshold, verbose=False)[0]
            b = r.boxes
            return [{"label": r.names[int(c)], "score": float(s), "box": _box(*xyxy)}
                    for xyxy, s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]
        return yolo

    if key == "mask2former":
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        mid = "facebook/mask2former-swin-large-coco-instance"
        proc = AutoImageProcessor.from_pretrained(mid)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(mid).to(_device()).eval()

        @torch.inference_mode()
        def m2f(image_path: str, threshold: float, classes=None) -> List[dict]:
            img = Image.open(image_path).convert("RGB")
            out = model(**proc(images=img, return_tensors="pt").to(model.device))
            res = proc.post_process_instance_segmentation(
                out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
            seg = res["segmentation"]
            dets = []
            for info in res["segments_info"]:
                ys, xs = torch.nonzero(seg == info["id"], as_tuple=True)
                box = _box(xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else {}
                dets.append({"label": model.config.id2label[info["label_id"]],
                             "score": float(info["score"]), "box": box})
            return dets
        return m2f

    if key in ("detr_r50", "detr_r101"):
        from transformers import pipeline
        mid = {"detr_r50": "facebook/detr-resnet-50", "detr_r101": "facebook/detr-resnet-101"}[key]
        pipe = pipeline("object-detection", model=mid, device=0 if _device() == "cuda" else -1)

        def detr(image_path: str, threshold: float, classes=None) -> List[dict]:
            return pipe(image_path, threshold=threshold)
        return detr

    if key == "deta_swin_large":
        import torch
        from PIL import Image
        try:
            from transformers import AutoImageProcessor, DetaForObjectDetection
        except ImportError as e:  # DETA is deprecated in recent transformers releases
            raise ImportError("DETA needs an older transformers release that still includes DetaForObjectDetection") from e
        mid = "jozhang97/deta-swin-large"
        proc = AutoImageProcessor.from_pretrained(mid)
        model = DetaForObjectDetection.from_pretrained(mid).to(_device()).eval()

        @torch.inference_mode()
        def deta(image_path: str, threshold: float, classes=None) -> List[dict]:
            img = Image.open(image_path).convert("RGB")
            out = model(**proc(images=img, return_tensors="pt").to(model.device))
            res = proc.post_process_object_detection(out, threshold=threshold,
                                                     target_sizes=torch.tensor([img.size[::-1]]))[0]
            return [{"label": model.config.id2label[int(l)], "score": float(s), "box": _box(*b.tolist())}
                    for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]
        return deta

    if key == "qwen3_vl_8b":
        return _qwen3_vl_counter("Qwen/Qwen3-VL-8B-Instruct")

    raise AssertionError(key)


# ------------------------------------------------------------------ VLM counter

def vlm_question(classes: Sequence[str]) -> Tuple[str, Dict[str, str]]:
    """The counting question for a VLM, and the map from answer key to canonical name."""
    from .classes import prompt_noun
    keys = {prompt_noun(c, 2): c for c in classes}
    listed = ", ".join(f'"{k}"' for k in keys)
    q = ("Count the objects in this image. For each object type below, count every instance "
         "you can see, including partly hidden ones. Count 0 if there are none.\n"
         f"Object types: {listed}\n"
         "Answer with only a JSON object mapping each object type to an integer, using exactly "
         "these keys, for example {" + ", ".join(f'"{k}": 0' for k in keys) + "}.")
    return q, keys


def parse_vlm_counts(text: str, keys: Mapping[str, str]) -> Optional[Dict[str, int]]:
    """Parse a VLM answer into {canonical name: count}. None if it is not usable."""
    import json
    import re
    from .classes import canonicalize
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out: Dict[str, int] = {}
    for k, v in obj.items():
        name = keys.get(k) or keys.get(str(k).strip().lower()) or canonicalize(str(k))
        if name is None or name not in keys.values():
            continue
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        if n < 0:
            return None
        out[name] = n
    if set(out) != set(keys.values()):
        return None
    return out


def counts_to_dets(counts: Mapping[str, int]) -> List[dict]:
    return [{"label": name, "score": 1.0, "box": {}} for name, n in counts.items() for _ in range(n)]


def _qwen3_vl_counter(model_id: str) -> Counter:
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _Auto
    except ImportError:  # older transformers
        from transformers import Qwen3VLForConditionalGeneration as _Auto
    model = _Auto.from_pretrained(model_id, dtype=torch.bfloat16, device_map=_device()).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def ask(image_path: str, question: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path},
                                                 {"type": "text", "text": question}]}]
        inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def vlm(image_path: str, threshold: float = 0.5, classes=None) -> List[dict]:
        if not classes:
            raise ValueError("the VLM counter needs the requested classes")
        q, keys = vlm_question(classes)
        text = ask(image_path, q)
        counts = parse_vlm_counts(text, keys)
        if counts is None:
            return [{"label": PARSE_ERROR, "score": 0.0, "box": {}, "raw": text[:300]}]
        return counts_to_dets(counts)
    return vlm
