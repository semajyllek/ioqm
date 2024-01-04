
from typing import Dict, Optional, Set
import numpy as np
from pathlib import Path
from itertools import combinations
import os

def add_label_combos(image_labels, label_combos):
  for l1, l2 in combinations(image_labels, 2):
    label_combos[l1, l2] += 1
    label_combos[l2, l1] += 1
  return label_combos

def get_coco_combos(coco_label_folder: Path, save_path: Optional[Path] = None) -> np.ndarray:
  label_combos = np.zeros((80, 80))
  for label_file in os.listdir(coco_label_folder):
    image_labels: Set[int] = ()
    with open(coco_label_folder / label_file, 'r') as f:
      for line in f.readlines():
        label = line.split(' ')[0]
        image_labels.add(int(label))
                
    label_combos = add_label_combos(image_labels, label_combos)

  if save_path is not None:
      np.save(save_path, label_combos)

  return label_combos


def get_class_examples(coco_label_folder) -> Dict[int, str]:
  to_find = {i: None for i in np.arange(80)}
  for label_file in os.listdir(coco_label_folder):
      with open(coco_label_folder / label_file, 'r') as f:
          for line in f.readlines():
              label = int(line.split(' ')[0])
              if to_find[label] is None:
                to_find[label] = label_file
  return to_find
                
        
if __name__ == "__main__":
  coco_label_folder = Path("/content/drive/MyDrive/coco/labels/train2017")
  save_path = Path("/content/drive/MyDrive/ioqm_data/label_combos.npy")
  get_coco_combos(coco_label_folder, save_path)

