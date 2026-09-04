"""Create argparse options for config files."""

import argparse
import json
import os.path as osp
from typing import Any

from easydict import EasyDict as edict

# keys every adv patch config must define, see adv_patch_gen/configs/README.md
REQUIRED_KEYS = {
    "augment_image",
    "batch_size",
    "class_list",
    "debug_mode",
    "device",
    "image_dir",
    "label_dir",
    "log_dir",
    "loss_target",
    "max_labels",
    "min_pixel_area",
    "min_tv_loss",
    "model_in_sz",
    "mul_gau_mean",
    "mul_gau_std",
    "n_classes",
    "n_epochs",
    "nps_mult",
    "objective_class_id",
    "patch_alpha",
    "patch_img_mode",
    "patch_name",
    "patch_pixel_range",
    "patch_save_epoch_freq",
    "patch_size",
    "patch_src",
    "random_patch_loc",
    "rotate_patches",
    "sal_mult",
    "start_lr",
    "target_size_frac",
    "tensorboard_batch_log_interval",
    "tensorboard_port",
    "transform_patches",
    "triplet_printfile",
    "tv_mult",
    "use_amp",
    "use_even_odd_images",
    "use_mul_add_gau",
    "val_epoch_freq",
    "val_image_dir",
    "weights_file",
    "x_off_loc",
    "y_off_loc",
}
OPTIONAL_KEYS = {"seed", "run_tensorboard"}
LOSS_TARGETS = {"obj", "cls", "obj * cls", "obj*cls"}
PATCH_IMG_MODES = {"L", "RGB"}


def _check_range(cfg: edict, key: str, errors: list) -> None:
    """A scalar or a two value [lo, hi] range, as accepted by np.random.uniform."""
    val = cfg[key]
    if isinstance(val, (int, float)):
        return
    if not (isinstance(val, (list, tuple)) and len(val) == 2 and val[0] <= val[1]):
        errors.append(f"{key} must be a number or an ascending [lo, hi] pair, got {val}")


def _check_pair(cfg: edict, key: str, errors: list, positive: bool = False) -> None:
    """A two value [height, width] style pair."""
    val = cfg[key]
    if not (isinstance(val, (list, tuple)) and len(val) == 2):
        errors.append(f"{key} must have exactly two values, got {val}")
    elif positive and not all(isinstance(v, int) and v > 0 for v in val):
        errors.append(f"{key} must be two positive ints, got {val}")


def validate_config(cfg: edict) -> edict:
    """Validate an adv patch config, raising ValueError listing every problem found."""
    errors = []
    unknown = set(cfg) - REQUIRED_KEYS - OPTIONAL_KEYS
    missing = REQUIRED_KEYS - set(cfg)
    if unknown:
        errors.append(f"unknown key(s): {sorted(unknown)}")
    if missing:
        errors.append(f"missing key(s): {sorted(missing)}")
    if missing:  # the checks below would just raise KeyError
        raise ValueError("Invalid config:\n  " + "\n  ".join(errors))

    if len(cfg.class_list) != cfg.n_classes:
        errors.append(f"n_classes ({cfg.n_classes}) != len(class_list) ({len(cfg.class_list)})")
    if cfg.patch_img_mode not in PATCH_IMG_MODES:
        errors.append(f"patch_img_mode must be one of {sorted(PATCH_IMG_MODES)}, got {cfg.patch_img_mode}")
    if cfg.loss_target not in LOSS_TARGETS:
        errors.append(f"loss_target must be one of {sorted(LOSS_TARGETS)}, got {cfg.loss_target}")
    if cfg.use_even_odd_images not in {"all", "even", "odd"}:
        errors.append(f"use_even_odd_images must be all, even or odd, got {cfg.use_even_odd_images}")
    if cfg.objective_class_id is not None and not 0 <= cfg.objective_class_id < cfg.n_classes:
        errors.append(f"objective_class_id must be null or in [0, {cfg.n_classes}), got {cfg.objective_class_id}")
    if not 0 < cfg.patch_alpha <= 1:
        errors.append(f"patch_alpha must be in (0, 1], got {cfg.patch_alpha}")
    if cfg.patch_src not in {"gray", "random"} and not osp.isfile(cfg.patch_src):
        errors.append(f'patch_src must be "gray", "random" or a path to an existing image, got {cfg.patch_src}')

    _check_pair(cfg, "model_in_sz", errors, positive=True)
    _check_pair(cfg, "patch_size", errors, positive=True)
    _check_pair(cfg, "x_off_loc", errors)
    _check_pair(cfg, "y_off_loc", errors)
    for key in ("target_size_frac", "mul_gau_mean", "mul_gau_std"):
        _check_range(cfg, key, errors)

    p_range = cfg.patch_pixel_range
    if not (isinstance(p_range, (list, tuple)) and len(p_range) == 2 and 0 <= p_range[0] < p_range[1] <= 255):
        errors.append(f"patch_pixel_range must be [lo, hi] within [0, 255] with lo < hi, got {p_range}")

    if errors:
        raise ValueError("Invalid config:\n  " + "\n  ".join(errors))
    return cfg


def load_config_object(cfg_path: str) -> edict:
    """Loads a config json, validates it and returns an edict object."""
    with open(cfg_path, "r", encoding="utf-8") as json_file:
        cfg_dict: Any = json.load(json_file)

    return validate_config(edict(cfg_dict))


def get_argparser(desc="Config file load for training adv patches") -> argparse.ArgumentParser:
    """Get parser with the default config argument."""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--cfg",
        "--config",
        type=str,
        dest="config",
        required=True,
        help="Path to JSON config file to use for adv patch generation (default: %(default)s)",
    )
    return parser
