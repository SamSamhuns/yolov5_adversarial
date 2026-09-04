"""Common utils."""

import random
import socket
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

IMG_EXTNS = {".png", ".jpg", ".jpeg"}


class BColors:
    """
    Border Color values for pretty printing in terminal
    Sample Use:
        print(f"{BColors.WARNING}Warning: Information.{BColors.ENDC}"
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def set_seed(seed: Optional[int]) -> None:
    """Seed python, numpy and torch RNGs for repeatability. A None seed leaves them untouched."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_port_in_use(port: int) -> bool:
    """Checks if a port is free for use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        return stream.connect_ex(("localhost", int(port))) == 0


def pad_to_square(img: Image, pad_rgb: Tuple[int, int, int] = (127, 127, 127)) -> Image:
    """Pads a PIL image to a square with pad_rgb values to the longest side."""
    w, h = img.size
    if w == h:
        padded_img = img
    else:
        if w < h:
            padding = (h - w) / 2
            padded_img = Image.new("RGB", (h, h), color=pad_rgb)
            padded_img.paste(img, (int(padding), 0))
        else:
            padding = (w - h) / 2
            padded_img = Image.new("RGB", (w, w), color=pad_rgb)
            padded_img.paste(img, (0, int(padding)))
    return padded_img


def calc_mean_and_std_err(arr: Union[list, np.ndarray]) -> Tuple[float, float]:
    """" Calculate mean and standard error."""
    mean = np.mean(arr)
    std_err = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return mean, std_err
