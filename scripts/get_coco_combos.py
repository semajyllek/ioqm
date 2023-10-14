
from typing import Optional
import numpy as np
from pathlib import Path
import os


def get_coco_combos(coco_label_folder: Path, save_path: Optional[Path] = None) -> np.ndarray:
	label_combos = np.zeros((80, 80))
	for label_file in os.listdir(coco_label_folder):
		image_labels = []
		with open(coco_label_folder / label_file, 'r') as f:
			for line in f.readlines():
				label = line.split(' ')[0]
				image_labels.append(int(label))
			
		for i in range(len(image_labels)):
			for j in range(i + 1, len(image_labels)):
				label_combos[image_labels[i]][image_labels[j]] += 1

	if save_path is not None:
		np.save(save_path, label_combos)

	return label_combos


if __name__ == "__main__":
	coco_label_folder = Path("/home/kevin/datasets/coco/labels/train2017")
	save_path = Path("/home/kevin/datasets/coco/labels/train2017/label_combos.npy")
	get_coco_combos(coco_label_folder, save_path)



