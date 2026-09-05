"""
Instructions for setting up with depthAI and OAK-D cameras
https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#first-steps-with-depthai.

requirements:
    depthai-sdk==1.9.4
"""

from __future__ import annotations

import time
from threading import Thread

import cv2
import depthai
import numpy as np
import onnxruntime
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import LOGGER, Profile, cv2, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

CLASS_LABELS = ["car", "truck", "bus", "people"]


class LoadOAKStream:
    # YOLOv5 OAK-D camera streamloader
    def __init__(self, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1, fps=25):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        self.img, self.frame, self.thread = None, 0, None

        cam_res = (img_size, img_size)
        self.pipeline = depthai.Pipeline()
        cam_rgb = self.pipeline.create(depthai.node.ColorCamera)
        cam_rgb.setPreviewSize(*cam_res)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(fps)

        xout_rgb = self.pipeline.create(depthai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        LOGGER.info(f"Cam FPS: {cam_rgb.getFps()}")
        # Start thread to read frames from video stream
        self.frame = float("inf")  # infinite stream fallback

        with depthai.Device(self.pipeline) as device:
            q_rgb = device.getOutputQueue("rgb")
            in_rgb = q_rgb.get()
            self.img = in_rgb.getCvFrame()
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in [self.img]])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning("WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.")

    def update(self):
        n, f = 0, self.frame
        with depthai.Device(self.pipeline) as device:
            while n < f:
                n += 1
                if n % self.vid_stride == 0:
                    q_rgb = device.getOutputQueue("rgb")
                    in_rgb = q_rgb.get()
                    if in_rgb:
                        im = in_rgb.getCvFrame()
                        self.img = im
                    else:
                        LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                        self.imgs = np.zeros_like(self.img)
                time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not self.thread.is_alive() or cv2.waitKey(2) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.img.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in [im0]])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return im, im0

    def __len__(self):
        return 1


def inference(
    weights: str = "runs/s_coco_e300_4Class_PeopleVehicle/weights/best.pt",
    device: str = "cpu",
    imgsz: tuple[int, int] = (640, 640),
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    classes: list | None = None,
    agnostic_nms: bool = False,
    half: bool = False,
    num_skip_frames: int = 0,
    cam_fps: int = 20,
    debug: bool = False,
    **kwrags,
) -> None:
    """Run Object Detection Application.

    Args:
        weights: str = path to yolov5 model
        device: str = cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz: Tuple[int, int] = inference size (height, width)
        conf_thres: float = confidence threshold
        iou_thres: float = NMS IOU threshold
        max_det: int = maximum detections per image
        classes: list = filter by class: --class 0, or --class 0 2 3
        agnostic_nms: bool = class-agnostic NMS
        half: bool = use half-precision
        num_skip_frames: int = num of frames to skip to speed processing
        cam_fps: int = only for oak-D camera
        debug: bool = prints fps info
    """
    # create depth ai pipeline
    pipeline = depthai.Pipeline()
    pipeline.setXLinkChunkSize(0)
    # set source & sink
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")

    # source properties
    cam_rgb.setPreviewSize(*imgsz)
    cam_rgb.setInterleaved(False)

    cam_rgb.preview.link(xout_rgb.input)
    cam_rgb.setFps(cam_fps)
    print(f"Cam FPS: {cam_rgb.getFps()}")

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)

    stride = 32
    auto = True
    names = model.names
    dt = (Profile(), Profile(), Profile())
    # Connect to device
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        counter = 0
        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                start = time.time()

                if counter % (num_skip_frames + 1) == 0:
                    im = letterbox(frame, imgsz[0], stride=stride, auto=auto)[0]  # padded resize
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)  # contiguous

                    with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt[1]:
                        pred = model(im)

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    im0 = frame
                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        annotator = Annotator(im0, line_width=1, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                # Add bbox to image
                                c = int(cls)  # integer class
                                label = f"{names[c]} {conf:.2f}"
                                annotator.box_label(xyxy, label, color=colors(c, True))

                        # Stream results
                        im0 = annotator.result()

                cv2.imshow("OAK", im0)

                end = time.time()
                inf_time = end - start
                fps = 1.0 / inf_time
                if debug:
                    print(f"Inference FPS: {fps:.2f} FPS")

                counter += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


def inference_threaded(
    weights: str = "runs/s_coco_e300_4Class_PeopleVehicle/weights/best.pt",
    device: str = "cpu",
    imgsz: tuple[int, int] = (640, 640),
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    classes: list | None = None,
    agnostic_nms: bool = False,
    half: bool = False,
    num_skip_frames: int = 0,
    cam_fps: int = 20,
    debug: bool = False,
    **kwrags,
) -> None:
    """Run Object Detection Application.

    Args:
        weights: str = path to yolov5 model
        device: str = cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz: Tuple[int, int] = inference size (height, width)
        conf_thres: float = confidence threshold
        iou_thres: float = NMS IOU threshold
        max_det: int = maximum detections per image
        classes: list = filter by class: --class 0, or --class 0 2 3
        agnostic_nms: bool = class-agnostic NMS
        half: bool = use half-precision
        num_skip_frames: int = num of frames to skip to speed processing
        cam_fps: int = only for oak-D camera
        debug: bool = prints fps info
    """
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)

    names = model.names
    stream = LoadOAKStream(img_size=640, stride=32, auto=True, fps=cam_fps)
    dt = (Profile(), Profile(), Profile())
    counter = 0
    for im, im0 in stream:
        start = time.time()

        if counter % (num_skip_frames + 1) == 0:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = model(im)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(im0, line_width=1, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()

            cv2.imshow("OAK", cv2.resize(im0, (960, 960)))

            end = time.time()
            inf_time = end - start
            fps = 1.0 / inf_time
            if debug:
                print(f"Inference FPS: {fps:.2f} FPS")

            counter += 1

    cv2.destroyAllWindows()


def inference_threaded_with_defense(
    weights: str = "runs/s_coco_e300_4Class_PeopleVehicle/weights/best.pt",
    def_weights: str = "runs/defendern250.onnx",
    device: str = "cpu",
    disp_res: tuple[int, int] = (960, 960),
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    classes: list | None = None,
    agnostic_nms: bool = False,
    half: bool = False,
    num_skip_frames: int = 0,
    cam_fps: int = 20,
    debug: bool = False,
) -> None:
    """Run Object Detection Application.

    Args:
        weights: str = path to yolov5 model
        def_weights: str = path to onnx autoencoder defense model
        device: str = cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz: Tuple[int, int] = inference size (height, width)
        conf_thres: float = confidence threshold
        iou_thres: float = NMS IOU threshold
        max_det: int = maximum detections per image
        classes: list = filter by class: --class 0, or --class 0 2 3
        agnostic_nms: bool = class-agnostic NMS
        half: bool = use half-precision
        num_skip_frames: int = num of frames to skip to speed processing
        cam_fps: int = only for oak-D camera
        debug: bool = prints fps info
    """
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)

    names = model.names
    stream = LoadOAKStream(img_size=640, stride=32, auto=True, fps=cam_fps)
    dt = (Profile(), Profile(), Profile())

    onnx_session = onnxruntime.InferenceSession(def_weights, None)
    onnx_input_name = onnx_session.get_inputs()[0].name

    counter = 0
    for im, im0 in stream:
        start = time.time()

        if counter % (num_skip_frames + 1) == 0:
            im_def = np.transpose(im.copy(), (0, 2, 3, 1)).astype(np.float32) / 255.0  # BCHW to BHWC
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference defense
            im_def = 2.0 * im_def - 1.0  # [0, 1] to [-1, 1]
            updates = onnx_session.run([], {onnx_input_name: im_def})[0] * 2.0  # get def updates
            im_def = np.clip(updates + im_def, -1.0, 1.0)
            im_def = (im_def + 1.0) / 2.0  # [-1, 1] to [0, 1]

            im_def = np.transpose(im_def, (0, 3, 1, 2))  # BHWC to BCHW
            im_def = torch.from_numpy(im_def).to(model.device)

            # Inference
            with dt[1]:
                pred = model(im)
                pred_def = model(im_def)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                pred_def = non_max_suppression(pred_def, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(im0, line_width=1, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()

            im02 = (im_def.cpu().numpy().transpose((0, 2, 3, 1)) * 255.0).astype(np.uint8)[0]  # BCHW to BHWC
            for i, det in enumerate(pred_def):  # per image
                annotator = Annotator(im02, line_width=1, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im02 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im02.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im02 = annotator.result()
            im02 = im02[:, :, ::-1]

            cv2.imshow("OAK attack", cv2.resize(im0, disp_res))
            cv2.imshow("OAK defense", cv2.resize(im02, disp_res))

            end = time.time()
            inf_time = end - start
            fps = 1.0 / inf_time
            if debug:
                print(f"Inference FPS: {fps:.2f} FPS")

            counter += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # "weights": "yolov5s.pt", # to use the base coco 80 class model
    kwargs = {
        "weights": "runs/s_coco_e300_4Class_PeopleVehicle/weights/best.pt",
        "def_weights": "runs/defendern250.onnx",
        "device": "cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        "conf_thres": 0.5,  # confidence threshold
        "max_det": 2,  # maximum detections per image
        "classes": [0],  # filter by class: --class 0, or --class 0 2 3
        "num_skip_frames": 0,  # num of frames to skip to speed processing
        "cam_fps": 20,  # only for oak-D camera
        "debug": False,  # prints fps info
    }

    # inference(**kwargs)
    # inference_threaded(**kwargs)
    inference_threaded_with_defense(**kwargs)
