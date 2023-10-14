
from transformers import AutoImageProcessor, DetaForObjectDetection
from sahi.predict import ObjectPrediction, visualize_object_predictions
import numpy as np
import PIL.Image
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_ID = 'jozhang97/deta-swin-large'
THRESHOLD = 0.2

image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
deta_model = DetaForObjectDetection.from_pretrained(MODEL_ID)
deta_model.to(device)


@torch.inference_mode()
def swin_detector(image_path: str, threshold: float = THRESHOLD) -> np.ndarray:
    image = PIL.Image.open(image_path)
    inputs = image_processor(images=image, return_tensors='pt').to(device)
    outputs = deta_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes)[0]

    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    cat_ids = results['labels'].cpu().numpy().tolist()
    preds = []
    for box, score, cat_id in zip(boxes, scores, cat_ids):
        box = np.round(box).astype(int)
        cat_label = deta_model.config.id2label[cat_id]
        pred = ObjectPrediction(bbox=box,
                                category_id=cat_id,
                                category_name=cat_label,
                                score=score)
        preds.append(pred)
    return preds