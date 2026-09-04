"""Detector loading for adversarial patch generation and testing.

Kept here rather than in models/common.py so the ultralytics yolov5 tree stays untouched.
"""

from typing import Optional

import torch

from models.common import DetectMultiBackend


def unwrap_preds(output):
    """Return the raw prediction tensor from a detector output.

    yolov5 in eval mode returns (inference_out, per_layer_out) and DetectMultiBackend hands that
    back as a list, while single output backends (onnx, tflite, ...) return a bare tensor. Indexing
    [0] unconditionally would slice the batch dimension for the latter.
    """
    return output[0] if isinstance(output, (list, tuple)) else output


class UltralyticsBackend(torch.nn.Module):
    """Thin wrapper exposing an ultralytics YOLO (v8/v11/...) model like DetectMultiBackend.

    The head emits [batch, 4 + n_classes, n_preds] with no objectness channel, which
    MaxProbExtractor detects and handles.
    """

    def __init__(self, weights: str, device: torch.device):
        super().__init__()
        from ultralytics import YOLO  # noqa: PLC0415  optional path, only needed for v8+ weights

        yolo = YOLO(weights)
        self.model = yolo.model.to(device).float().eval()
        self.names = getattr(yolo, "names", getattr(self.model, "names", {}))
        self.device = device
        self.pt = True  # torch backend, so patch training can backprop through it

    def forward(self, im, augment=False):
        """Run the wrapped model, returning its raw prediction output."""
        return self.model(im)


def load_detector(weights: str, device: torch.device, backend: Optional[str] = None) -> torch.nn.Module:
    """Load the detector the patch is optimized against.

    Args:
        weights: path to the detector weights
        device: torch device to place the model on
        backend: "yolov5" for this repo's DetectMultiBackend (all its export formats), "ultralytics"
            for a v8/v11 checkpoint loaded through the ultralytics package. None means yolov5.

    Note: only torch backends can be used for patch *training*. Every exported format in
    DetectMultiBackend returns predictions via torch.from_numpy, which severs the autograd graph,
    so no gradient can reach the patch. They are usable for testing an existing patch.
    """
    backend = backend or "yolov5"
    if backend == "ultralytics":
        return UltralyticsBackend(weights, device)
    if backend == "yolov5":
        return DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False).eval()
    raise ValueError(f'model_backend must be "yolov5" or "ultralytics", got {backend}')


def supports_patch_training(model: torch.nn.Module) -> bool:
    """Whether gradients can reach the patch through this model.

    Exported backends (onnx, engine, openvino, tflite, ...) round-trip through numpy inside
    DetectMultiBackend.forward, so they cannot be trained against.
    """
    return bool(getattr(model, "pt", False) or getattr(model, "jit", False))
