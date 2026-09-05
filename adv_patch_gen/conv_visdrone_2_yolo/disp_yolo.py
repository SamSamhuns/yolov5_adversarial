import argparse
import sys

sys.path.append(".")

import cv2

from adv_patch_gen.conv_visdrone_2_yolo.utils import get_annot_img_paths, load_yolo_annots_as_np

# visdrone dataset classes
# ignore(0), pedestrian(1), people(2), bicycle(3),
# car(4), van(5), truck(6), tricycle(7), awning-tricycle(8),
# bus(9), motor(10), others(11)
# CLASS_IDS_2_CONSIDER = {4, 5, 6, 9}

ANNOT_EXT = {".txt"}
IMG_EXT = {".jpg", ".png"}
# custom yolo classes
# car(0), van(1), truck(2), bus(3)
CLASS_IDS_2_CONSIDER = {0, 1, 2, 3}


def get_parsed_args():
    parser = argparse.ArgumentParser(description="Disp YOLO annotated images")
    parser.add_argument(
        "-a",
        "--source_annot_dir",
        type=str,
        dest="source_annot_dir",
        required=True,
        help="YOLO annotation source dir. Should contain annot txt files (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--source_image_dir",
        type=str,
        dest="source_image_dir",
        required=True,
        help="YOLO images source dir. Should contain image files (default: %(default)s)",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_parsed_args()
    annot_paths, image_paths = get_annot_img_paths(args.source_annot_dir, args.source_image_dir, ANNOT_EXT, IMG_EXT)

    for ant, img in zip(annot_paths, image_paths):
        image = cv2.imread(img)
        annots = load_yolo_annots_as_np(ant)
        for annot in annots:
            class_id = annot[0]
            if class_id not in CLASS_IDS_2_CONSIDER:  # car, van ,truck, bus
                continue
            ih, iw, _ = image.shape
            xc, yc, w, h = annot[1] * iw, annot[2] * ih, annot[3] * iw, annot[4] * ih
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow("YOLO annot visualized", image)
        key = cv2.waitKey()
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()


if __name__ == "__main__":
    main()
