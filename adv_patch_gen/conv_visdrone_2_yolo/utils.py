from __future__ import annotations

import glob
import os

import numpy as np


def get_annot_img_paths(
    annot_dir: str, image_dir: str, annot_ext: set[str], img_ext: set[str]
) -> tuple[list[str], list[str]]:
    annots_path = os.path.join(annot_dir, "*")
    images_path = os.path.join(image_dir, "*")
    annot_paths = [p for p in sorted(glob.glob(annots_path)) if os.path.splitext(p)[-1] in annot_ext]
    image_paths = [p for p in sorted(glob.glob(images_path)) if os.path.splitext(p)[-1] in img_ext]

    assert len(annot_paths) == len(image_paths)
    return annot_paths, image_paths


def load_visdrone_annots_as_np(annot_file: str) -> np.ndarray:
    annot_list = []
    with open(annot_file) as f:
        for values in f:
            annots = list(map(int, values.strip().strip(",").split(",")))
            x1, y1 = annots[0], annots[1]
            x2, y2 = x1 + annots[2], y1 + annots[3]
            score, class_id, occu = annots[4], annots[5], annots[7]
            annot_list.append([x1, y1, x2, y2, score, class_id, occu])
    return np.asarray(annot_list)


def load_yolo_annots_as_np(annot_file: str) -> np.ndarray:
    annot_list = []
    with open(annot_file) as f:
        for values in f:
            annots = list(map(float, values.strip().split()))
            class_id = annots[0]
            xc, yc = annots[1], annots[2]
            w, h = annots[3], annots[4]
            annot_list.append([class_id, xc, yc, w, h])
    return np.asarray(annot_list)
