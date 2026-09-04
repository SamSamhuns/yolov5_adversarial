"""Adversarial patch augmentation for detector training (the defense side of the paper).

BboxPatcher pastes patches from a directory onto ground truth boxes while training a detector, so
the detector sees patched objects during training. It is driven by train.py --patch_dir and the
bbox_patch probability in the hyp yaml.

Originally added to utils/augmentations.py in 516357fe, where a later merge with ultralytics master
dropped it while leaving every call site intact, breaking `import utils.dataloaders` and therefore
train.py, val.py, detect.py and export.py. It lives here so the yolov5 tree only needs the import
line rather than carrying ~100 lines of fork code that a future merge can drop again.
"""

import glob
import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

PATCH_EXTNS = {".jpeg", ".jpg", ".png"}


class BboxPatcher:
    """Paste randomly transformed patches onto ground truth boxes of a training image."""

    def __init__(
        self,
        patch_dir: str,
        rotation_range: Tuple[float, float] = (-20, 20),
        scale_range: Tuple[float, float] = (0.10, 0.30),
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        m_gau_mean: Tuple[float, float] = (0.7, 0.9),
        m_gau_std: Tuple[float, float] = (0.1, 0.1),
        patch_apply_prob: float = 0.5,
    ):
        """
        Args:
            patch_dir: dir with patch images. An empty or missing dir leaves self.patches empty,
                which callers check before applying.
            rotation_range: rotation range in degrees
            scale_range: patch area as a fraction of the bbox area
            patch_apply_prob: probability that any one box receives a patch
        """
        self.patches = []
        for patch_path in sorted(glob.glob(os.path.join(patch_dir, "*"))) if patch_dir else []:
            if os.path.splitext(patch_path)[1].lower() in PATCH_EXTNS:
                patch = cv2.imread(patch_path)
                if patch is not None:
                    self.patches.append(TF.to_tensor(patch))
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_factor = 0.01
        self.m_gau_mean = m_gau_mean
        self.m_gau_std = m_gau_std
        self.patch_apply_prob = patch_apply_prob

    def __call__(self, image: np.ndarray, bbox_coords: np.ndarray) -> np.ndarray:
        """
        Arguments:
            image: np.ndarray, image of shape H,W,C
            bbox_coords: np.ndarray, [[cls,x1,y1,x2,y2], ...]
        """
        if not self.patches or len(bbox_coords) == 0:
            return image
        img_h, img_w = image.shape[:2]
        image = TF.to_tensor(image)

        for bbox in bbox_coords:
            # apply to this box with probability patch_apply_prob
            if np.random.random() >= self.patch_apply_prob:
                continue
            patch = self.patches[np.random.randint(0, len(self.patches))]
            # add and mul with gaussian noise
            p_c, p_h, p_w = patch.shape
            mul_gau = torch.normal(
                np.random.uniform(*self.m_gau_mean), np.random.uniform(*self.m_gau_std), (p_c, p_h, p_w)
            )
            add_gau = torch.normal(0, 0.001, (p_c, p_h, p_w))
            patch = patch * mul_gau + add_gau

            # adjust brightness, contrast & add uniform noise
            patch = TF.adjust_brightness(patch, np.random.uniform(*self.brightness_range))
            patch = TF.adjust_contrast(patch, np.random.uniform(*self.contrast_range))
            patch = patch + torch.empty_like(patch).uniform_(-1, 1) * self.noise_factor

            # Randomly select rotation and scale parameters
            rotation = np.random.uniform(*self.rotation_range)
            scale = np.random.uniform(*self.scale_range)

            _, x1, y1, x2, y2 = map(int, bbox)
            # Calculate the width and height of the bounding box
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            # square patch sized to cover `scale` of the bbox area
            psize = max(int((bbox_width * bbox_height * scale) ** (1 / 2)), 1)
            patch = TF.resize(patch, [psize, psize], antialias=True)

            # create patch mask, then rotate patch and mask together
            patch_mask = torch.ones_like(patch)
            patch = TF.rotate(patch, rotation, expand=True, fill=0)
            patch_mask = TF.rotate(patch_mask, rotation, expand=True, fill=0)
            patch_scaled_h, patch_scaled_w = patch.shape[1:]  # expanded rotation changes size

            # Calculate the position to place the patch at the center of the bbox
            x_pos = int(x1 + (bbox_width - patch_scaled_w) / 2)
            y_pos = int(y1 + (bbox_height - patch_scaled_h) / 2)
            pad = (x_pos, y_pos, img_w - (x_pos + patch_scaled_w), img_h - (y_pos + patch_scaled_h))
            # a patch hanging off the image edge would need negative padding, which TF.pad rejects
            if min(pad) < 0:
                continue
            padded_patch = TF.pad(patch, list(pad))
            padded_patch_mask = TF.pad(patch_mask, list(pad))

            # Apply the patch to the image
            image = image * (1 - padded_patch_mask) + padded_patch * padded_patch_mask

        return np.asarray(T.ToPILImage()(image.clamp(0, 1)))
