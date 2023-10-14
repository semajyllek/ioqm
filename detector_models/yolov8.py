
from ultralytics import YOLO
from sahi.predict import ObjectPrediction

import numpy as np
import PIL.Image

THRESHOLD = 0.5

model_yolo = YOLO('yolov8x.pt')

def yolov8_detector(image_path: str, threshold: float = THRESHOLD) -> np.ndarray:
    image = PIL.Image.open(image_path)
    results = model_yolo(image, imgsz=640, conf=threshold)
    boxes = results[0].boxes.cpu().numpy().data
    preds = []
    for box in boxes:
        score = box[4]
        cat_id = int(box[5])
        box = np.round(box[:4]).astype(int)
        cat_label = model_yolo.model.names[cat_id]
        pred = ObjectPrediction(bbox=box,
                                category_id=cat_id,
                                category_name=cat_label,
                                score=score)
        preds.append(pred)

    return preds