import argparse
import sys

sys.path.append(".")

import cv2

from adv_patch_gen.conv_visdrone_2_yolo.utils import get_annot_img_paths, load_visdrone_annots_as_np

# visdrone dataset classes
# ignore(0), pedestrian(1), people(2), bicycle(3),
# car(4), van(5), truck(6), tricycle(7), awning-tricycle(8),
# bus(9), motor(10), others(11)

ANNOT_EXT = {".txt"}
IMG_EXT = {".jpg", ".png"}
CLASS_IDS_2_CONSIDER = {4, 5, 6, 9}


def get_parsed_args():
    parser = argparse.ArgumentParser(description="Disp VisDrone annotated images")
    parser.add_argument(
        "-a",
        "--source_annot_dir",
        type=str,
        dest="source_annot_dir",
        required=True,
        help="VisDrone annotation source dir. Should contain annot txt files (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--source_image_dir",
        type=str,
        dest="source_image_dir",
        required=True,
        help="VisDrone images source dir. Should contain image files (default: %(default)s)",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_parsed_args()
    annot_paths, image_paths = get_annot_img_paths(args.source_annot_dir, args.source_image_dir, ANNOT_EXT, IMG_EXT)

    for ant, img in zip(annot_paths, image_paths):
        image = cv2.imread(img)
        annots = load_visdrone_annots_as_np(ant)
        for annot in annots:
            score, class_id = annot[4], annot[5]
            if class_id not in CLASS_IDS_2_CONSIDER:  # car, van ,truck, bus
                continue
            color = (0, 0, 255) if score == 0 else (0, 255, 0)
            x1, y1, x2, y2 = annot[:4]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

        cv2.imshow("VisDrone annot visualized", image)
        key = cv2.waitKey()
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()


if __name__ == "__main__":
    main()
