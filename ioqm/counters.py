"""
Counters (detectors) used as evaluators or best-of-N verifiers.

Every counter is a callable `counter(image_path, threshold) -> list[dict]`,
each dict being {"label", "score", "box": {xmin, ymin, xmax, ymax}}. That is the
Hugging Face pipeline format, which ioqm.detect.normalize_detections accepts.
Labels are canonicalized downstream (ioqm.classes), so the older names
Mask2Former uses ("aeroplane", "tvmonitor", ...) map correctly.

DEFAULT_THRESHOLDS are pre-calibration starting points. Phase 1 replaces them
with per-class thresholds fitted to human labels. Detections are stored at a
low floor, so changing thresholds never requires re-running a counter.

Heavy imports are lazy, so `import ioqm.counters` works without torch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List

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
}
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

        def yolo(image_path: str, threshold: float) -> List[dict]:
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
        def m2f(image_path: str, threshold: float) -> List[dict]:
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

        def detr(image_path: str, threshold: float) -> List[dict]:
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
        def deta(image_path: str, threshold: float) -> List[dict]:
            img = Image.open(image_path).convert("RGB")
            out = model(**proc(images=img, return_tensors="pt").to(model.device))
            res = proc.post_process_object_detection(out, threshold=threshold,
                                                     target_sizes=torch.tensor([img.size[::-1]]))[0]
            return [{"label": model.config.id2label[int(l)], "score": float(s), "box": _box(*b.tolist())}
                    for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]
        return deta

    raise AssertionError(key)
