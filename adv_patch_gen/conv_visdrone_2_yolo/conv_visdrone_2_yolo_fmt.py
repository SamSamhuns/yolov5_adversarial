"""
Convert VisDrone annotation format to YOLO labels format Works for training of YOLOv5 and YOLOv7.

YOLOv7 requires an additional txt file (Same name as the first parent directory) with paths to the images for the train,
val & test splits
"""

import argparse
import glob
import os
import os.path as osp
from pathlib import Path
from typing import Optional

import imagesize
import tqdm

# VisDrone annot fmt
# <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
#
# classes:
#   ignore(0), pedestrian(1), people(2), bicycle(3),
#   car(4), van(5), truck(6), tricycle(7),
#   awning-tricycle(8), bus(9), motor(10), others(11)

# YOLO annot fmt
# One row per object
# Each row is class x_center y_center width height format.
# Box coordinates must be in normalized xywh format (from 0 - 1).
# If boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
# Class numbers are zero-indexed (start from 0).


# only get pedestrian, people, car, van, truck, bus classes from VisDrone
CLASS_2_CONSIDER = {1, 2, 4, 5, 6, 9}
# optionally remap class ids, can be set to None
CLASS_ID_REMAP = {4: 0, 5: 0, 6: 1, 9: 2, 1: 3, 2: 3}
IMG_EXT = {".jpg", ".png"}


def get_parsed_args():
    parser = argparse.ArgumentParser(description="VisDrone to YOLO annot format")
    parser.add_argument(
        "--sad",
        "--source_annot_dir",
        type=str,
        dest="source_annot_dir",
        required=True,
        help="VisDrone annotation source dir. Should contain annot txt files (default: %(default)s)",
    )
    parser.add_argument(
        "--sid",
        "--source_image_dir",
        type=str,
        dest="source_image_dir",
        required=True,
        help="VisDrone images source dir. Should contain image files (default: %(default)s)",
    )
    parser.add_argument(
        "--td",
        "--target_annot_dir",
        type=str,
        dest="target_annot_dir",
        required=True,
        help="YOLO annotation target dir. YOLO by default uses dirname labels (default: %(default)s)",
    )
    parser.add_argument(
        "--dc",
        "--low_dim_cutoff",
        type=int,
        dest="low_dim_cutoff",
        default=None,
        help="All bboxes with dims(w/h) < cutoff pixel are ignored i.e 400 (default: %(default)s)",
    )
    parser.add_argument(
        "--ac",
        "--low_area_cutoff",
        type=float,
        dest="low_area_cutoff",
        default=None,
        help="All bboxes with area perc < cutoff area perc are ignored i.e. 0.01 (default: %(default)s)",
    )
    args = parser.parse_args()
    return args


def conv_visdrone_2_yolo(
    source_annot_dir: str,
    source_image_dir: str,
    target_annot_dir: str,
    low_dim_cutoff: Optional[int],
    low_area_cutoff: Optional[float],
):
    """
    low_dim_cutoff: int, lower cutoff for bounding boxes width/height dims in pixels
    low_area_cutoff: float, lower area perc cutoff for bounding box areas in perc
    """
    if not all([osp.isdir(source_annot_dir), osp.isdir(source_image_dir)]):
        raise ValueError(
            f"source_annot_dir({source_annot_dir}) and source_image_dir({source_image_dir}) must be directories"
        )
    src_annot_path = osp.join(source_annot_dir, "*")
    src_image_path = osp.join(source_image_dir, "*")
    src_annot_paths = sorted(glob.glob(src_annot_path))
    src_image_paths = [p for p in sorted(glob.glob(src_image_path)) if osp.splitext(p)[-1] in IMG_EXT]
    assert len(src_image_paths) == len(
        src_annot_paths
    ), f"Num src images: {len(src_image_paths)} & num src annots: {len(src_annot_paths)} do not match"

    os.makedirs(target_annot_dir, exist_ok=True)
    low_dim_cutoff = float("-inf") if not low_dim_cutoff else low_dim_cutoff
    low_area_cutoff = float("-inf") if not low_area_cutoff else low_area_cutoff
    split_name = Path(source_annot_dir).resolve().parent.name.lower()
    target_img_list_fpath = osp.join(osp.dirname(target_annot_dir), split_name + ".txt")

    with tqdm.tqdm(total=len(src_image_paths)) as pbar, open(target_img_list_fpath, "w", encoding="utf-8") as imgw:
        orig_box_count = new_box_count = skipped = 0
        for src_annot_file, src_image_file in zip(src_annot_paths, src_image_paths):
            try:
                iw, ih = imagesize.get(src_image_file)
                target_annot_file = osp.join(target_annot_dir, osp.basename(src_annot_file))
                with open(src_annot_file, "r", encoding="utf-8") as fr, open(
                    target_annot_file, "w", encoding="utf-8"
                ) as fw:
                    for coords in fr:
                        annots = list(map(int, coords.strip().strip(",").split(",")))
                        x, y = annots[0], annots[1]
                        w, h = annots[2], annots[3]
                        score, class_id, occu = annots[4], annots[5], annots[7]
                        if class_id not in CLASS_2_CONSIDER:  # only keep classes to consider
                            continue
                        orig_box_count += 1
                        if w < low_dim_cutoff or h < low_dim_cutoff:  # cutoff value for dims to remove outliers
                            continue
                        area_perc = 100 * (w * h) / (iw * ih)
                        if area_perc < low_area_cutoff:  # cutoff value for area perc to remove outliers
                            continue

                        xc, yc = x + (w / 2), y + (h / 2)
                        # only use objects used for eval along and all levels of occlusion (0,1,2)
                        if score and occu <= 2:
                            class_id = CLASS_ID_REMAP[class_id] if CLASS_ID_REMAP else class_id
                            fw.write(f"{class_id} {xc/iw} {yc/ih} {w/iw} {h/ih}\n")
                            new_box_count += 1
                target_image_path = osp.join(
                    osp.dirname(osp.dirname(target_annot_file)),
                    "images",
                    osp.basename(target_annot_file).split(".")[0] + osp.splitext(src_image_file)[1],
                )
                imgw.write(f"{osp.abspath(target_image_path)}\n")
            except (OSError, ValueError, IndexError, KeyError) as excep:
                # a malformed annot line or unreadable image, keep going but do not hide it
                skipped += 1
                print(f"{type(excep).__name__}: {excep}. Skipping {src_image_file}")
            pbar.update(1)
        if skipped:
            print(f"Skipped {skipped} of {len(src_image_paths)} images, see the errors above")
        print(f"Original Box Count: {orig_box_count}. Converted Box Count {new_box_count}")
        if orig_box_count:
            print(f"{100 * new_box_count / orig_box_count:.2f}% of total boxes kept")
        else:
            print(f"No boxes of classes {sorted(CLASS_2_CONSIDER)} found in {source_annot_dir}")


def main():
    args = get_parsed_args()
    print(args)
    conv_visdrone_2_yolo(
        args.source_annot_dir, args.source_image_dir, args.target_annot_dir, args.low_dim_cutoff, args.low_area_cutoff
    )


if __name__ == "__main__":
    main()
