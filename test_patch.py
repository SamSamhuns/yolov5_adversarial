"""Testing code for evaluating Adversarial patches against object detection."""

import glob
import io
import json
import os
import os.path as osp
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm
from easydict import EasyDict as edict
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms

from adv_patch_gen.utils.common import IMG_EXTNS, BColors, pad_to_square, set_seed
from adv_patch_gen.utils.config_parser import get_argparser, load_config_object
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from adv_patch_gen.utils.video import (
    ffmpeg_combine_three_vids,
    ffmpeg_combine_two_vids,
    ffmpeg_create_video_from_image_dir,
)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, xyxy2xywh
from utils.metrics import ConfusionMatrix
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

torch.backends.cudnn.benchmark = False


def eval_coco_metrics(anno_json: str, pred_json: str, txt_save_path: str, w_mode: str = "a") -> np.ndarray:
    """Compare and eval pred json producing coco metrics."""
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    evaluator = COCOeval(anno, pred, "bbox")

    evaluator.evaluate()
    evaluator.accumulate()

    # capture evaluator stats once, then print and save them
    std_out = io.StringIO()
    with redirect_stdout(std_out):
        evaluator.summarize()
    eval_stats = std_out.getvalue()
    print(eval_stats)
    with open(txt_save_path, w_mode, encoding="utf-8") as fwriter:
        fwriter.write(eval_stats)
    return evaluator.stats


def conf_matrix_tp_fp(conf_matrix: ConfusionMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """Per class true and false positives from a confusion matrix, excluding the background class."""
    tp = conf_matrix.matrix.diagonal()
    fp = conf_matrix.matrix.sum(1) - tp
    return tp[:-1], fp[:-1]


def plot_conf_matrix(conf_matrix: ConfusionMatrix, save_dir: str, class_list: List[str], save_name: str) -> None:
    """Plot a confusion matrix to save_dir/save_name. ConfusionMatrix.plot() always writes confusion_matrix.png."""
    conf_matrix.plot(save_dir=save_dir, names=class_list)
    default_path = osp.join(save_dir, "confusion_matrix.png")
    if osp.isfile(default_path):
        os.replace(default_path, osp.join(save_dir, save_name))


class PatchTester:
    """Module for testing patches on dataset against object detection models."""

    def __init__(self, cfg: edict) -> None:
        self.cfg = cfg
        self.dev = select_device(cfg.device)

        model = DetectMultiBackend(cfg.weights_file, device=self.dev, dnn=False, data=None, fp16=False)
        self.model = model.eval().to(self.dev)
        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, self.dev
        ).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)

    @staticmethod
    def calc_asr(
        boxes,
        boxes_pred,
        class_list: List[str],
        conf_thresh: float = 0.25,
        lo_area: float = 20**2,
        hi_area: float = 67**2,
        cls_id: Optional[int] = None,
        class_agnostic: bool = False,
        recompute_asr_all: bool = False,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate attack success rate (How many bounding boxes were hidden from the detector) for all predictions and
        for different bbox areas.

        Note cls_id is None, misclassifications are ignored and only missing detections are considered attack success.
        Args:
            boxes: torch.Tensor, first pass boxes (gt unpatched boxes) [class, x1, y1, x2, y2]
            boxes_pred: torch.Tensor, second pass boxes (patched boxes) [x1, y1, x2, y2, conf, class]
            class_list: list of class names in correct order
            conf_thresh: confidence threshold used for the detections, forwarded to the confusion matrix
            lo_area: small bbox area threshold
            hi_area: large bbox area threshold
            cls_id: filter for a particular class
            class_agnostic: All classes are considered the same
            recompute_asr_all: Recomputer ASR for all boxes aggregated together slower but more acc. asr
        Return:
            attack success rates bbox area tuple: small, medium, large, all
                float, float, float, float
        """
        # if cls_id is provided and evaluation is not class agnostic then mis-clsfs count as attack success
        if cls_id is not None:
            boxes = boxes[boxes[:, 0] == cls_id]
            boxes_pred = boxes_pred[boxes_pred[:, 5] == cls_id]

        boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        boxes_pred_area = (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])

        b_small = boxes[boxes_area < lo_area]
        bp_small = boxes_pred[boxes_pred_area < lo_area]
        b_med = boxes[torch.logical_and(boxes_area <= hi_area, boxes_area >= lo_area)]
        bp_med = boxes_pred[torch.logical_and(boxes_pred_area <= hi_area, boxes_pred_area >= lo_area)]
        b_large = boxes[boxes_area > hi_area]
        bp_large = boxes_pred[boxes_pred_area > hi_area]
        assert (bp_small.shape[0] + bp_med.shape[0] + bp_large.shape[0]) == boxes_pred.shape[0]
        assert (b_small.shape[0] + b_med.shape[0] + b_large.shape[0]) == boxes.shape[0]

        conf_matrix = ConfusionMatrix(len(class_list), conf=conf_thresh)
        conf_matrix.process_batch(bp_small, b_small)
        tps_small, fps_small = conf_matrix_tp_fp(conf_matrix)
        conf_matrix = ConfusionMatrix(len(class_list), conf=conf_thresh)
        conf_matrix.process_batch(bp_med, b_med)
        tps_med, fps_med = conf_matrix_tp_fp(conf_matrix)
        conf_matrix = ConfusionMatrix(len(class_list), conf=conf_thresh)
        conf_matrix.process_batch(bp_large, b_large)
        tps_large, fps_large = conf_matrix_tp_fp(conf_matrix)
        if recompute_asr_all:
            conf_matrix = ConfusionMatrix(len(class_list), conf=conf_thresh)
            conf_matrix.process_batch(boxes_pred, boxes)
            tps_all, fps_all = conf_matrix_tp_fp(conf_matrix)
        else:
            tps_all, fps_all = tps_small + tps_med + tps_large, fps_small + fps_med + fps_large

        # class agnostic mode (Mis-clsfs are ignored, only non-dets matter)
        if class_agnostic:
            tp_small = tps_small.sum() + fps_small.sum()
            tp_med = tps_med.sum() + fps_med.sum()
            tp_large = tps_large.sum() + fps_large.sum()
            tp_all = tps_all.sum() + fps_all.sum()
        # filtering by cls_id or non class_agnostic mode (Mis-clsfs are successes)
        elif cls_id is not None:  # consider single class, mis-clsfs or non-dets
            tp_small = tps_small[cls_id]
            tp_med = tps_med[cls_id]
            tp_large = tps_large[cls_id]
            tp_all = tps_all[cls_id]
        else:  # non class_agnostic, mis-clsfs or non-dets
            tp_small = tps_small.sum()
            tp_med = tps_med.sum()
            tp_large = tps_large.sum()
            tp_all = tps_all.sum()

        asr_small = 1.0 - tp_small / (b_small.shape[0] + 1e-6)
        asr_medium = 1.0 - tp_med / (b_med.shape[0] + 1e-6)
        asr_large = 1.0 - tp_large / (b_large.shape[0] + 1e-6)
        asr_all = 1.0 - tp_all / (boxes.shape[0] + 1e-6)

        return max(asr_small, 0.0), max(asr_medium, 0.0), max(asr_large, 0.0), max(asr_all, 0.0)

    @staticmethod
    def draw_bbox_on_pil_image(bbox: np.ndarray, padded_img_pil: Image, class_list: List[str]) -> Image:
        """Draw bounding box on a PIL image and return said image after drawing."""
        padded_img_np = np.ascontiguousarray(padded_img_pil)
        label_2_class = dict(enumerate(class_list))

        annotator = Annotator(padded_img_np, line_width=1, example=str(label_2_class))
        for *xyxy, conf, cls in bbox:
            cls_int = int(cls)  # integer class
            label = f"{label_2_class[cls_int]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(cls_int, True))
        return Image.fromarray(padded_img_np)

    def _create_coco_image_annot(self, file_path: Path, width: int, height: int, image_id: int) -> dict:
        file_path = file_path.name
        image_annotation = {
            "file_name": file_path,
            "height": height,
            "width": width,
            "id": image_id,
        }
        return image_annotation

    def test(
        self,
        conf_thresh: float = 0.4,
        nms_thresh: float = 0.4,
        save_txt: bool = False,
        save_image: bool = False,
        save_orig_padded_image: bool = True,
        draw_bbox_on_image: bool = True,
        class_agnostic: bool = False,
        cls_id: Optional[int] = None,
        min_pixel_area: Optional[int] = None,
        save_plots: bool = False,
        save_video: bool = False,
        max_images: int = 100000,
        apply_patch_transforms: bool = True,
    ) -> dict:
        """
        Initiate test for properly, randomly and no-patched images
        Args:
            conf_thresh: confidence thres for successful detection/positives
            nms_thresh: nms thres
            save_txt: save the txt yolo format detections for the clean, properly and randomly patched images
            save_image: save properly and randomly patched images
            save_orig_padded_image: save orig padded images
            draw_bbox_on_image: Draw bboxes on the original images and the random noise & properly patched images
            class_agnostic: all classes are treated the same. Use when only evaluating for obj det & not classification
            cls_id: filtering for a specific class for evaluation only
            min_pixel_area: all bounding boxes having area less than this are filtered out during testing. if None, use all boxes
            save_video: if set to true, eval videos are saved in directory videos
            max_images: max number of images to evaluate from inside imgdir
            apply_patch_transforms: apply rotation, location shift, brightness and contrast transforms to the patch
        Returns:
            dict of patch and noise coco_map and asr results
        """
        t_0 = time.time()

        patch_size = self.cfg.patch_size
        model_in_sz = self.cfg.model_in_sz
        m_h, m_w = model_in_sz

        patch_img = Image.open(self.cfg.patchfile).convert(self.cfg.patch_img_mode)
        patch_img = transforms.Resize(patch_size)(patch_img)
        adv_patch_cpu = transforms.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.dev)

        # make dirs
        clean_img_dir = osp.join(self.cfg.savedir, "clean/", "images/")
        clean_txt_dir = osp.join(self.cfg.savedir, "clean/", "labels/")
        proper_img_dir = osp.join(self.cfg.savedir, "proper_patched/", "images/")
        proper_txt_dir = osp.join(self.cfg.savedir, "proper_patched/", "labels/")
        random_img_dir = osp.join(self.cfg.savedir, "random_patched/", "images/")
        random_txt_dir = osp.join(self.cfg.savedir, "random_patched/", "labels/")
        jsondir = osp.join(self.cfg.savedir, "results_json")
        video_dir = osp.join(self.cfg.savedir, "videos")

        print(f"Saving all outputs to {self.cfg.savedir}")
        dirs_to_create = [jsondir]
        if save_txt:
            dirs_to_create.extend([clean_txt_dir, proper_txt_dir, random_txt_dir])
        if save_image:
            dirs_to_create.extend([proper_img_dir, random_img_dir])
        if save_image and save_orig_padded_image:
            dirs_to_create.append(clean_img_dir)
        if save_image and save_video:
            dirs_to_create.append(video_dir)
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)

        # dump cfg json file to self.cfg.savedir
        with open(osp.join(self.cfg.savedir, "cfg.json"), "w", encoding="utf-8") as f_json:
            json.dump(self.cfg, f_json, ensure_ascii=False, indent=4)

        # save patch to self.cfg.savedir
        patch_save_path = osp.join(self.cfg.savedir, self.cfg.patchfile.split("/")[-1])
        transforms.ToPILImage(self.cfg.patch_img_mode)(adv_patch_cpu).save(patch_save_path)

        img_paths = glob.glob(osp.join(self.cfg.imgdir, "*"))
        img_paths = sorted([p for p in img_paths if osp.splitext(p)[-1] in IMG_EXTNS])

        print("Total num images:", len(img_paths))
        img_paths = img_paths[:max_images]
        print("Considered num images:", len(img_paths))

        # stores json results
        clean_gt_results = []
        clean_results = []
        noise_results = []
        patch_results = []

        clean_image_annotations = []
        # to calc confusion matrixes and attack success rates later
        all_labels = []
        all_patch_preds = []
        all_noise_preds = []
        det_boxes = dropped_boxes = 0

        #### iterate through all images ####
        box_id = 0
        transforms_resize = transforms.Resize(model_in_sz)
        transforms_totensor = transforms.ToTensor()
        transforms_topil = transforms.ToPILImage("RGB")
        zeros_tensor = torch.zeros([1, 5]).to(self.dev)
        for imgfile in tqdm.tqdm(img_paths):
            img_name = osp.splitext(imgfile)[0].split("/")[-1]
            imgfile_path = Path(imgfile)
            image_id = int(imgfile_path.stem) if imgfile_path.stem.isnumeric() else imgfile_path.stem

            clean_image_annotation = self._create_coco_image_annot(
                imgfile_path, width=m_w, height=m_h, image_id=image_id
            )
            clean_image_annotations.append(clean_image_annotation)

            txtname = img_name + ".txt"
            txtpath = osp.join(clean_txt_dir, txtname)
            # open image and adjust to yolo input size
            padded_img_pil = pad_to_square(Image.open(imgfile).convert("RGB"))
            padded_img_pil = transforms_resize(padded_img_pil)

            #######################################
            # generate labels to use later for patched image
            padded_img_tensor = transforms_totensor(padded_img_pil).unsqueeze(0).to(self.dev)
            with torch.no_grad():
                pred = self.model(padded_img_tensor)
                boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            count_before_drop = boxes.shape[0]
            det_boxes += count_before_drop
            # filter det bounding boxes by pixel area
            if min_pixel_area is not None:
                boxes = boxes[((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) > min_pixel_area]
            dropped_boxes += count_before_drop - boxes.shape[0]
            all_labels.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            labels = []
            if save_txt:
                textfile = open(txtpath, "w+", encoding="utf-8")
            for box in boxes:
                cls_id_box = box[-1].item()
                score = box[4].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                labels.append([cls_id_box, x_center / m_w, y_center / m_h, width / m_w, height / m_h])
                if save_txt:
                    textfile.write(f"{cls_id_box} {x_center/m_w} {y_center/m_h} {width/m_w} {height/m_h}\n")
                clean_results.append(
                    {
                        "image_id": image_id,
                        "bbox": [x_center - width / 2, y_center - height / 2, width, height],
                        "score": round(score, 5),
                        "category_id": 0 if class_agnostic else int(cls_id_box),
                    }
                )
                clean_gt_results.append(
                    {
                        "id": box_id,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [x_center - width / 2, y_center - height / 2, width, height],
                        "area": width * height,
                        "category_id": 0 if class_agnostic else int(cls_id_box),
                        "segmentation": [],
                    }
                )
                box_id += 1
            if save_txt:
                textfile.close()

            # save img
            cleanname = img_name + ".jpg"
            if save_image and save_orig_padded_image:
                if draw_bbox_on_image:
                    padded_img_drawn = PatchTester.draw_bbox_on_pil_image(
                        all_labels[-1], padded_img_pil, self.cfg.class_list
                    )
                    padded_img_drawn.save(osp.join(clean_img_dir, cleanname))
                else:
                    padded_img_pil.save(osp.join(clean_img_dir, cleanname))

            # use a filler zeros array for no dets
            label = np.asarray(labels) if labels else np.zeros([1, 5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            #######################################
            # Apply proper patches
            img_fake_batch = padded_img_tensor
            lab_fake_batch = label.unsqueeze(0).to(self.dev)
            if len(lab_fake_batch[0]) == 1 and torch.equal(lab_fake_batch[0], zeros_tensor):
                # no det, use images without patches
                p_tensor_batch = padded_img_tensor
            else:
                # transform patch and add it to image
                adv_batch_t = self.patch_transformer(
                    adv_patch,
                    lab_fake_batch,
                    model_in_sz,
                    use_mul_add_gau=apply_patch_transforms,
                    do_transforms=apply_patch_transforms,
                    do_rotate=apply_patch_transforms,
                    rand_loc=apply_patch_transforms,
                )
                p_tensor_batch = self.patch_applier(img_fake_batch, adv_batch_t)

            properpatchedname = img_name + ".jpg"
            # generate a label file for the image with sticker
            txtname = properpatchedname.replace(".jpg", ".txt")
            txtpath = osp.join(proper_txt_dir, txtname)

            with torch.no_grad():
                pred = self.model(p_tensor_batch)
                boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            # filter det bounding boxes by pixel area
            if min_pixel_area is not None:
                boxes = boxes[((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) > min_pixel_area]
            all_patch_preds.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            if save_txt:
                textfile = open(txtpath, "w+", encoding="utf-8")
            for box in boxes:
                cls_id_box = box[-1].item()
                score = box[4].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                if save_txt:
                    textfile.write(f"{cls_id_box} {x_center/m_w} {y_center/m_h} {width/m_w} {height/m_h}\n")
                patch_results.append(
                    {
                        "image_id": image_id,
                        "bbox": [x_center - (width / 2), y_center - (height / 2), width, height],
                        "score": round(score, 5),
                        "category_id": 0 if class_agnostic else int(cls_id_box),
                    }
                )
            if save_txt:
                textfile.close()

            # save properly patched img
            if save_image:
                p_img_pil = transforms_topil(p_tensor_batch.squeeze(0).cpu())
                if draw_bbox_on_image:
                    p_img_pil_drawn = PatchTester.draw_bbox_on_pil_image(
                        all_patch_preds[-1], p_img_pil, self.cfg.class_list
                    )
                    p_img_pil_drawn.save(osp.join(proper_img_dir, properpatchedname))
                else:
                    p_img_pil.save(osp.join(proper_img_dir, properpatchedname))

            #######################################
            # Apply random patches
            if len(lab_fake_batch[0]) == 1 and torch.equal(lab_fake_batch[0], zeros_tensor):
                # no det, use images without patches
                p_tensor_batch = padded_img_tensor
            else:
                # create a random patch, transform it and add it to image
                random_patch = torch.rand(adv_patch_cpu.size()).to(self.dev)
                adv_batch_t = self.patch_transformer(
                    random_patch,
                    lab_fake_batch,
                    model_in_sz,
                    use_mul_add_gau=apply_patch_transforms,
                    do_transforms=apply_patch_transforms,
                    do_rotate=apply_patch_transforms,
                    rand_loc=apply_patch_transforms,
                )
                p_tensor_batch = self.patch_applier(img_fake_batch, adv_batch_t)

            randompatchedname = img_name + ".jpg"
            # generate a label file for the image with random patch
            txtname = randompatchedname.replace(".jpg", ".txt")
            txtpath = osp.join(random_txt_dir, txtname)

            with torch.no_grad():
                pred = self.model(p_tensor_batch)
                boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            # filter det bounding boxes by pixel area
            if min_pixel_area is not None:
                boxes = boxes[((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) > min_pixel_area]
            all_noise_preds.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            if save_txt:
                textfile = open(txtpath, "w+", encoding="utf-8")
            for box in boxes:
                cls_id_box = box[-1].item()
                score = box[4].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                if save_txt:
                    textfile.write(f"{cls_id_box} {x_center/m_w} {y_center/m_h} {width/m_w} {height/m_h}\n")
                noise_results.append(
                    {
                        "image_id": image_id,
                        "bbox": [x_center - (width / 2), y_center - (height / 2), width, height],
                        "score": round(score, 5),
                        "category_id": 0 if class_agnostic else int(cls_id_box),
                    }
                )
            if save_txt:
                textfile.close()

            # save randomly patched img
            if save_image:
                p_img_pil = transforms_topil(p_tensor_batch.squeeze(0).cpu())
                if draw_bbox_on_image:
                    p_img_pil_drawn = PatchTester.draw_bbox_on_pil_image(
                        all_noise_preds[-1], p_img_pil, self.cfg.class_list
                    )
                    p_img_pil_drawn.save(osp.join(random_img_dir, randompatchedname))
                else:
                    p_img_pil.save(osp.join(random_img_dir, randompatchedname))

        torch.cuda.empty_cache()

        # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
        all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
        # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
        all_patch_preds = torch.cat(all_patch_preds)
        all_noise_preds = torch.cat(all_noise_preds)

        # Calc confusion matrices if not class_agnostic
        if not class_agnostic and save_plots:
            patch_confusion_matrix = ConfusionMatrix(len(self.cfg.class_list), conf=conf_thresh)
            patch_confusion_matrix.process_batch(all_patch_preds, all_labels)
            noise_confusion_matrix = ConfusionMatrix(len(self.cfg.class_list), conf=conf_thresh)
            noise_confusion_matrix.process_batch(all_noise_preds, all_labels)

            plot_conf_matrix(patch_confusion_matrix, self.cfg.savedir, self.cfg.class_list, "conf_matrix_patch.png")
            plot_conf_matrix(noise_confusion_matrix, self.cfg.savedir, self.cfg.class_list, "conf_matrix_noise.png")

        # add all required fields for a reference GT clean annotation
        clean_gt_results_json = {"annotations": clean_gt_results, "categories": [], "images": clean_image_annotations}
        for index, label in enumerate(self.cfg.class_list, start=0):
            categories = {"supercategory": "Defect", "id": index, "name": label}
            clean_gt_results_json["categories"].append(categories)

        # save all json results
        clean_gt_json = osp.join(jsondir, "clean_gt_results.json")
        clean_json = osp.join(jsondir, "clean_results.json")
        noise_json = osp.join(jsondir, "noise_results.json")
        patch_json = osp.join(jsondir, "patch_results.json")

        with open(clean_gt_json, "w", encoding="utf-8") as f_json:
            json.dump(clean_gt_results_json, f_json, ensure_ascii=False, indent=4)
        with open(clean_json, "w", encoding="utf-8") as f_json:
            json.dump(clean_results, f_json, ensure_ascii=False, indent=4)
        with open(noise_json, "w", encoding="utf-8") as f_json:
            json.dump(noise_results, f_json, ensure_ascii=False, indent=4)
        with open(patch_json, "w", encoding="utf-8") as f_json:
            json.dump(patch_results, f_json, ensure_ascii=False, indent=4)

        patch_txt_path = osp.join(self.cfg.savedir, "patch_map_stats.txt")
        noise_txt_path = osp.join(self.cfg.savedir, "noise_map_stats.txt")
        clean_txt_path = osp.join(self.cfg.savedir, "clean_map_stats.txt")

        print(f"{BColors.HEADER}### Metrics for images with no patches for baseline. Should be ~1 ###{BColors.ENDC}")
        eval_coco_metrics(clean_gt_json, clean_json, clean_txt_path)

        print(f"{BColors.HEADER}### Metrics for images with correct patches ###{BColors.ENDC}")
        coco_map_patch = eval_coco_metrics(clean_gt_json, patch_json, patch_txt_path) if patch_results else []

        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(
            all_labels,
            all_patch_preds,
            self.cfg.class_list,
            conf_thresh=conf_thresh,
            cls_id=cls_id,
            class_agnostic=class_agnostic,
        )
        with open(patch_txt_path, "a", encoding="utf-8") as f_patch:
            asr_str = ""
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= small | = {asr_s:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=medium | = {asr_m:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= large | = {asr_l:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=   all | = {asr_a:.3f}\n"
            print(asr_str)
            f_patch.write(asr_str + "\n")
        metrics_patch = {"coco_map": coco_map_patch, "asr": [asr_s, asr_m, asr_l, asr_a]}

        print(f"{BColors.HEADER}### Metrics for images with random noise patches ###{BColors.ENDC}")
        coco_map_noise = eval_coco_metrics(clean_gt_json, noise_json, noise_txt_path) if noise_results else []

        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(
            all_labels,
            all_noise_preds,
            self.cfg.class_list,
            conf_thresh=conf_thresh,
            cls_id=cls_id,
            class_agnostic=class_agnostic,
        )
        with open(noise_txt_path, "a", encoding="utf-8") as f_noise:
            asr_str = ""
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= small | = {asr_s:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=medium | = {asr_m:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= large | = {asr_l:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=   all | = {asr_a:.3f}\n"
            print(asr_str)
            f_noise.write(asr_str + "\n")
        metrics_noise = {"coco_map": coco_map_noise, "asr": [asr_s, asr_m, asr_l, asr_a]} if noise_results else []

        if save_image and save_video:
            patch_vid = osp.join(video_dir, "patch.mp4")
            random_vid = osp.join(video_dir, "random.mp4")
            clean_vid = osp.join(video_dir, "clean.mp4")
            ffmpeg_create_video_from_image_dir(proper_img_dir, patch_vid, title_text="Adversarial Patch")
            ffmpeg_create_video_from_image_dir(random_img_dir, random_vid, title_text="Random Patch")
            ffmpeg_create_video_from_image_dir(clean_img_dir, clean_vid, title_text="No Patch")
            ffmpeg_combine_two_vids(clean_vid, patch_vid, osp.join(video_dir, "clean_patch.mp4"))
            ffmpeg_combine_two_vids(clean_vid, random_vid, osp.join(video_dir, "clean_random.mp4"))
            ffmpeg_combine_three_vids(clean_vid, random_vid, patch_vid, osp.join(video_dir, "clean_random_patch.mp4"))

        if min_pixel_area:
            dperc = 100 * dropped_boxes / det_boxes
            drop_str = f" Det Boxes: {det_boxes} | Dropped Boxes: {dropped_boxes} | Dropped: {dperc:.2f}% @gt{min_pixel_area} px\n"
            print(drop_str)
            with open(clean_txt_path, "a", encoding="utf-8") as fwriter:
                fwriter.write(drop_str)
        t_f = time.time()
        print(f" Time to complete evaluation = {t_f - t_0} seconds")
        return {"patch": metrics_patch, "noise": metrics_noise}


def main():
    parser = get_argparser(
        desc="Test patches on a directory with images. Params from argparse take precedence over those from config"
    )
    parser.add_argument(
        "--dev",
        type=str,
        dest="device",
        default=None,
        required=False,
        help='Device to use (i.e. cpu, cuda:0). If None, use "device" in cfg json (default: %(default)s)',
    )
    parser.add_argument(
        "--ts",
        "--target_size_frac",
        type=float,
        nargs="+",
        dest="target_size_frac",
        default=None,
        required=False,
        help="Patch target_size_frac of the bbox area. Two values set a range. "
        'If unset, use "target_size_frac" in cfg json (default: %(default)s)',
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        dest="weights",
        default=None,
        required=False,
        help='Path to yolov5 model wt. If None, use "weights_file" path in cfg json (default: %(default)s)',
    )
    parser.add_argument(
        "-p",
        "--patchfile",
        type=str,
        dest="patchfile",
        default=None,
        required=True,
        help="Path to patch image file for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--id",
        "--imgdir",
        type=str,
        dest="imgdir",
        default=None,
        required=True,
        help="Path to img dir for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--sd",
        "--savedir",
        type=str,
        dest="savedir",
        default="runs/test_adversarial",
        help="Path to save dir for saving testing results (default: %(default)s)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        dest="conf_thresh",
        default=0.4,
        required=False,
        help="Conf threshold for detection (default: %(default)s)",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        dest="nms_thresh",
        default=0.4,
        required=False,
        help="IoU threshold for NMS (default: %(default)s)",
    )
    parser.add_argument(
        "--no-patch-transforms",
        dest="no_patch_transforms",
        action="store_true",
        help="Disable the random rotation, location shift, brightness and contrast transforms on the patch",
    )
    parser.add_argument(
        "--save-txt",
        dest="savetxt",
        action="store_true",
        help="Save txt files with predicted labels in yolo fmt for later inspection",
    )
    parser.add_argument(
        "--save-img", dest="saveimg", action="store_true", help="Save images with patches for later inspection"
    )
    parser.add_argument(
        "--save-vid",
        dest="savevideo",
        action="store_true",
        help="Combine no-patch, random-patch and proper-patched images into videos",
    )
    parser.add_argument(
        "--save-plot", dest="saveplots", action="store_true", help="Save the patch and noise confusion matrix plots"
    )
    parser.add_argument(
        "--class-agnostic",
        dest="class_agnostic",
        action="store_true",
        help="All classes are treated the same. Use when only evaluating for obj det & not clsf",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        dest="target_class",
        default=None,
        required=False,
        help="Target specific class with id for misclassification test (default: %(default)s)",
    )
    parser.add_argument(
        "--min-pixel-area",
        type=int,
        dest="min_pixel_area",
        default=None,
        required=False,
        help="all bboxes having area < this are filtered in test. if None, use all bboxes (default: %(default)s)",
    )

    args = parser.parse_args()
    cfg = load_config_object(args.config)
    set_seed(cfg.get("seed", 42))
    cfg.device = args.device if args.device is not None else cfg.device
    cfg.weights_file = (
        args.weights if args.weights is not None else cfg.weights_file
    )  # check if cfg.weights_file is ignored
    cfg.patchfile = args.patchfile
    cfg.imgdir = args.imgdir
    if args.target_size_frac is not None:
        if len(args.target_size_frac) not in {1, 2}:
            raise ValueError("target_size_frac can only have one or two values")
        cfg.target_size_frac = args.target_size_frac[0] if len(args.target_size_frac) == 1 else args.target_size_frac

    if args.savevideo and not args.saveimg:
        raise ValueError("To save videos, images must also be saved pass both --save-img & --save-vid flags")
    savename = f'{time.strftime("%Y%m%d-%H%M%S")}_' + cfg.patch_name
    if args.class_agnostic and args.target_class is not None:
        print(
            f"""{BColors.WARNING}WARNING:{BColors.ENDC} target_class and class_agnostic are both set.
              Target_class will be ignored and metrics will be class agnostic. Only set either."""
        )
        args.target_class = None
    else:
        savename += f"_tc{args.target_class}" if args.target_class is not None else ""
    savename += "_agnostic" if args.class_agnostic else ""
    savename += f"_gt{args.min_pixel_area}" if args.min_pixel_area is not None else ""
    cfg.savedir = osp.join(args.savedir, savename)

    print(f"{BColors.OKBLUE} Test Arguments: {args} {BColors.ENDC}")
    tester = PatchTester(cfg)
    tester.test(
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        save_txt=args.savetxt,
        save_image=args.saveimg,
        class_agnostic=args.class_agnostic,
        cls_id=args.target_class,
        min_pixel_area=args.min_pixel_area,
        save_plots=args.saveplots,
        save_video=args.savevideo,
        apply_patch_transforms=not args.no_patch_transforms,
    )


if __name__ == "__main__":
    main()
