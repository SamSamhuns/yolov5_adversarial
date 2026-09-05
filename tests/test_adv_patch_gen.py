"""Tests for the adversarial patch generation code under adv_patch_gen/.

Run with: python -m pytest tests/test_adv_patch_gen.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("easydict")
pytest.importorskip("PIL")

import numpy as np
from easydict import EasyDict
from PIL import Image as PILImage

from adv_patch_gen.utils.config_parser import validate_config
from adv_patch_gen.utils.dataset import YOLODataset
from adv_patch_gen.utils.loss import MaxProbExtractor, NPSLoss, SaliencyLoss, TotalVariationLoss
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CFG = REPO_ROOT / "adv_patch_gen/configs/base.json"
TRIPLETS = REPO_ROOT / "adv_patch_gen/utils/30_rgb_triplets.csv"
DEV = torch.device("cpu")
MODEL_IN_SZ = (640, 640)


def make_transformer(**kwargs):
    """PatchTransformer with all randomness pinned unless a test overrides it."""
    opts = {
        "t_size_frac": 0.3,
        "mul_gau_mean": 0.5,
        "mul_gau_std": 0.1,
        "x_off_loc": [0.0, 0.0],
        "y_off_loc": [0.0, 0.0],
        "dev": DEV,
    }
    opts.update(kwargs)
    return PatchTransformer(**opts).to(DEV)


def transform(patch, lab, **fwd):
    """Apply the transformer with every stochastic knob off unless overridden."""
    opts = {
        "use_mul_add_gau": False,
        "do_transforms": False,
        "do_rotate": False,
        "rand_loc": False,
        "do_perspective": False,
    }
    opts.update({k: v for k, v in fwd.items() if k in opts})
    tf = make_transformer(**{k: v for k, v in fwd.items() if k not in opts})
    return tf(patch, lab, MODEL_IN_SZ, **opts)


def nonzero_bbox(plane):
    """Pixel bbox (x_min, y_min, x_max, y_max) of the non-zero region of a [C, H, W] tensor."""
    ys, xs = torch.nonzero(plane.abs().sum(0) > 0, as_tuple=True)
    return xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()


def labels(*boxes, max_labels=4):
    """Build a [1, max_labels, 5] label batch, zero padded like YOLODataset does."""
    lab = torch.zeros(1, max_labels, 5)
    for i, (cls, xc, yc, w, h) in enumerate(boxes):
        lab[0, i] = torch.tensor([cls, xc, yc, w, h])
    return lab


class TestPatchTransformer:
    """Geometry and numerical safety of the patch warp."""

    def test_patch_lands_centered_on_its_box(self):
        patch = torch.full((3, 64, 64), 0.5)
        out = transform(patch, labels((0, 0.5, 0.5, 0.2, 0.2)))
        assert out.shape == (1, 4, 3, 640, 640)

        x0, y0, x1, y1 = nonzero_bbox(out[0, 0])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        assert abs(cx - 320) <= 2 and abs(cy - 320) <= 2, f"patch centered at {(cx, cy)}, expected ~(320, 320)"

        # target_size = t_size_frac * sqrt((w*m_w)^2 + (h*m_h)^2)
        expected = 0.3 * ((0.2 * 640) ** 2 + (0.2 * 640) ** 2) ** 0.5
        assert abs((x1 - x0) - expected) <= 4, f"width {x1 - x0}, expected ~{expected:.1f}"
        assert abs((y1 - y0) - expected) <= 4, f"height {y1 - y0}, expected ~{expected:.1f}"

    def test_zero_area_filler_rows_produce_no_patch_and_no_nan(self):
        """Rows past the real labels are zero padding and must contribute nothing."""
        patch = torch.full((3, 64, 64), 0.5)
        out = transform(patch, labels((0, 0.5, 0.5, 0.2, 0.2)))
        assert torch.isfinite(out).all(), "filler rows used to divide by zero and put inf in theta"
        assert out[0, 0].abs().sum() > 0, "the real box should get a patch"
        for i in range(1, 4):
            assert out[0, i].abs().sum() == 0, f"filler row {i} should be empty"

    def test_all_filler_batch_is_empty(self):
        out = transform(torch.full((3, 64, 64), 0.5), labels())
        assert torch.isfinite(out).all()
        assert out.abs().sum() == 0

    def test_y_offset_uses_the_y_range(self):
        """Regression: off_y was drawn from x_off_loc, so y_off_loc never had any effect."""
        patch = torch.full((3, 64, 64), 0.5)
        lab = labels((0, 0.5, 0.5, 0.2, 0.2))
        base = transform(patch, lab)
        # a fixed +0.5 of box height downward, x pinned
        moved = transform(patch, lab, x_off_loc=[0.0, 0.0], y_off_loc=[0.5, 0.5], rand_loc=True)

        _, by0, _, by1 = nonzero_bbox(base[0, 0])
        _, my0, _, my1 = nonzero_bbox(moved[0, 0])
        shift = ((my0 + my1) / 2) - ((by0 + by1) / 2)
        expected = 0.5 * 0.2 * 640  # off_y = h * 0.5, in model input pixels
        assert abs(shift - expected) <= 3, f"y shifted by {shift}, expected ~{expected}"

    def test_x_offset_uses_the_x_range(self):
        patch = torch.full((3, 64, 64), 0.5)
        lab = labels((0, 0.5, 0.5, 0.2, 0.2))
        base = transform(patch, lab)
        moved = transform(patch, lab, x_off_loc=[0.5, 0.5], y_off_loc=[0.0, 0.0], rand_loc=True)

        bx0, _, bx1, _ = nonzero_bbox(base[0, 0])
        mx0, _, mx1, _ = nonzero_bbox(moved[0, 0])
        shift = ((mx0 + mx1) / 2) - ((bx0 + bx1) / 2)
        expected = 0.5 * 0.2 * 640
        assert abs(shift - expected) <= 3, f"x shifted by {shift}, expected ~{expected}"

    def test_patch_size_scales_with_box_size(self):
        patch = torch.full((3, 64, 64), 0.5)
        small = transform(patch, labels((0, 0.5, 0.5, 0.1, 0.1)))
        large = transform(patch, labels((0, 0.5, 0.5, 0.4, 0.4)))
        sx0, _, sx1, _ = nonzero_bbox(small[0, 0])
        lx0, _, lx1, _ = nonzero_bbox(large[0, 0])
        assert (lx1 - lx0) > 3 * (sx1 - sx0)

    def test_output_stays_in_pixel_range(self):
        out = transform(torch.full((3, 64, 64), 0.5), labels((0, 0.5, 0.5, 0.2, 0.2)))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_non_square_patch_keeps_its_aspect(self):
        out = transform(torch.full((3, 32, 64), 0.5), labels((0, 0.5, 0.5, 0.3, 0.3)))
        x0, y0, x1, y1 = nonzero_bbox(out[0, 0])
        assert abs((x1 - x0) / (y1 - y0) - 2.0) < 0.15, "a 2:1 patch should render about 2:1"


class TestPatchApplier:
    """The applier must only touch pixels the patch covers."""

    def test_patch_replaces_only_where_it_landed(self):
        img = torch.full((1, 3, 640, 640), 0.25)
        adv = transform(torch.full((3, 64, 64), 0.9), labels((0, 0.5, 0.5, 0.2, 0.2)))
        out = PatchApplier(1.0)(img, adv)

        covered = adv[0].abs().sum(0).sum(0) > 0  # [H, W]
        assert torch.allclose(out[0, :, ~covered], torch.tensor(0.25)), "background must be untouched"
        assert not torch.allclose(out[0, :, covered], torch.tensor(0.25)), "patch area must change"

    def test_alpha_blend_is_between_image_and_patch(self):
        img = torch.zeros(1, 3, 640, 640)
        adv = transform(torch.full((3, 64, 64), 1.0), labels((0, 0.5, 0.5, 0.2, 0.2)))
        blended = PatchApplier(0.5)(img, adv)
        covered = adv[0].abs().sum(0).sum(0) > 0
        assert blended[0, :, covered].max() <= 0.5 + 1e-5


class TestLosses:
    """Loss shapes, ranges and the sizes they must not be coupled to."""

    def test_nps_loss_works_for_any_patch_size(self):
        """Regression: the printability array was materialized at one fixed patch size."""
        nps = NPSLoss(str(TRIPLETS))
        for shape in [(3, 64, 64), (3, 32, 128), (3, 7, 5)]:
            val = nps(torch.rand(shape))
            assert val.ndim == 0 and torch.isfinite(val), f"bad nps for {shape}"

    def test_nps_loss_is_zero_for_a_printable_color(self):
        first = [float(v) for v in TRIPLETS.read_text().splitlines()[0].split(",")]
        patch = torch.zeros(3, 16, 16)
        for chan, val in enumerate(first):
            patch[chan] = val
        nps_val = NPSLoss(str(TRIPLETS))(patch)
        assert nps_val < 1e-3, f"a patch painted an exact printable color should score ~0, got {nps_val}"

    def test_saliency_loss_is_not_divided_by_patch_size(self):
        """Regression: dividing by numel made sal_mult a no-op at ~1e-5."""
        colorful = torch.zeros(3, 64, 64)
        colorful[0], colorful[1], colorful[2] = 1.0, 0.0, 0.0
        small = torch.zeros(3, 8, 8)
        small[0], small[1], small[2] = 1.0, 0.0, 0.0

        sal = SaliencyLoss()
        big_val, small_val = sal(colorful), sal(small)
        assert big_val > 0.1, f"colorfulness should be an absolute value in ~[0, 1], got {big_val}"
        assert torch.allclose(big_val, small_val, atol=1e-5), "colorfulness must not depend on patch size"

    def test_saliency_is_lower_for_a_gray_patch(self):
        sal = SaliencyLoss()
        gray = sal(torch.full((3, 64, 64), 0.5))
        colorful = torch.rand(3, 64, 64)
        colorful[0] *= 0.1
        assert gray < sal(colorful)

    def test_total_variation_is_near_zero_for_a_flat_patch(self):
        tv = TotalVariationLoss()
        flat = tv(torch.full((3, 64, 64), 0.5))
        noisy = tv(torch.rand(3, 64, 64))
        assert flat < 1e-3 and noisy > flat

    def test_max_prob_extractor_does_not_renormalize_class_scores(self):
        """Regression: a softmax was applied to scores yolov5 has already sigmoided."""
        n_classes = 4
        cfg = SimpleNamespace(n_classes=n_classes, objective_class_id=2, loss_target=lambda obj, cls: obj * cls)
        out = torch.zeros(1, 3, 5 + n_classes)
        out[..., 4] = 1.0  # objectness
        out[0, 1, 5 + 2] = 0.8  # target class score on one box
        assert torch.allclose(MaxProbExtractor(cfg)(out), torch.tensor([0.8]), atol=1e-6)

    def test_max_prob_extractor_takes_the_max_over_boxes(self):
        cfg = SimpleNamespace(n_classes=2, objective_class_id=None, loss_target=lambda obj, cls: obj * cls)
        out = torch.zeros(2, 5, 7)
        out[..., 4] = 1.0
        out[0, 3, 5] = 0.7
        out[1, 0, 6] = 0.4
        assert torch.allclose(MaxProbExtractor(cfg)(out), torch.tensor([0.7, 0.4]), atol=1e-6)

    def test_topk_of_one_is_the_original_max(self):
        cfg = SimpleNamespace(n_classes=2, objective_class_id=None, loss_target=lambda o, c: o * c, loss_topk=1)
        out = torch.zeros(1, 8, 7)
        out[..., 4] = 1.0
        out[0, :, 5] = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.05, 0.7, 0.2, 0.4])
        assert torch.allclose(MaxProbExtractor(cfg)(out), torch.tensor([0.9]), atol=1e-6)

    def test_topk_averages_the_k_best_predictions(self):
        """Regression: a single max sends gradient to 1 of ~25k predictions per image."""
        scores = [0.1, 0.9, 0.3, 0.8, 0.05, 0.7, 0.2, 0.4]
        out = torch.zeros(1, 8, 7)
        out[..., 4] = 1.0
        out[0, :, 5] = torch.tensor(scores)
        for k in (2, 3, 8):
            cfg = SimpleNamespace(n_classes=2, objective_class_id=None, loss_target=lambda o, c: o * c, loss_topk=k)
            expected = sum(sorted(scores, reverse=True)[:k]) / k
            assert torch.allclose(MaxProbExtractor(cfg)(out), torch.tensor([expected]), atol=1e-6), f"k={k}"

    def test_topk_larger_than_prediction_count_is_clamped(self):
        cfg = SimpleNamespace(n_classes=2, objective_class_id=None, loss_target=lambda o, c: o * c, loss_topk=999)
        out = torch.zeros(1, 3, 7)
        out[..., 4] = 1.0
        out[0, :, 5] = torch.tensor([0.3, 0.6, 0.9])
        assert torch.allclose(MaxProbExtractor(cfg)(out), torch.tensor([0.6]), atol=1e-6)

    def test_multiple_target_classes_take_the_best_of_them(self):
        cfg = SimpleNamespace(n_classes=4, objective_class_id=[1, 3], loss_target=lambda o, c: o * c, loss_topk=1)
        out = torch.zeros(1, 4, 9)
        out[..., 4] = 1.0
        out[0, 0, 5 + 0] = 0.95  # untargeted class, must be ignored
        out[0, 1, 5 + 1] = 0.60  # targeted
        out[0, 2, 5 + 3] = 0.75  # targeted, the best of the targeted ones
        assert torch.allclose(MaxProbExtractor(cfg)(out), torch.tensor([0.75]), atol=1e-6)

    def test_yolov8_style_head_is_detected(self):
        """v8/v11 emit [batch, 4 + n_classes, n_preds] with no objectness channel."""
        n_cls = 4
        cfg = SimpleNamespace(n_classes=n_cls, objective_class_id=2, loss_target=lambda o, c: o * c, loss_topk=1)
        out = torch.zeros(1, 4 + n_cls, 6)  # transposed layout
        out[0, 4 + 2, 3] = 0.83
        assert torch.allclose(MaxProbExtractor(cfg)(out), torch.tensor([0.83]), atol=1e-6)

    def test_unrecognized_head_shape_raises(self):
        cfg = SimpleNamespace(n_classes=4, objective_class_id=None, loss_target=lambda o, c: o * c, loss_topk=1)
        with pytest.raises(ValueError, match="Unrecognized detector output"):
            MaxProbExtractor(cfg)(torch.zeros(1, 11, 17))


class TestEOT:
    """The expectation-over-transformation knobs added for physical realism."""

    def test_perspective_off_reproduces_no_perspective(self):
        torch.manual_seed(0)
        base = transform(torch.full((3, 64, 64), 0.5), labels((0, 0.5, 0.5, 0.3, 0.3)), perspective_scale=0.0)
        torch.manual_seed(0)
        same = transform(
            torch.full((3, 64, 64), 0.5), labels((0, 0.5, 0.5, 0.3, 0.3)), perspective_scale=0.0, do_perspective=True
        )
        assert torch.allclose(base, same), "perspective_scale=0 must be a no-op"

    def test_perspective_warps_the_patch(self):
        patch = torch.full((3, 64, 64), 0.5)
        lab = labels((0, 0.5, 0.5, 0.3, 0.3))
        flat = transform(patch, lab, perspective_scale=0.0)
        tilted = transform(patch, lab, perspective_scale=0.4, do_perspective=True)
        assert torch.isfinite(tilted).all(), "the plane clamp must keep the grid finite"
        assert not torch.allclose(flat, tilted), "a non-zero perspective_scale should change the render"
        assert tilted.min() >= 0.0 and tilted.max() <= 1.0

    def test_illumination_ramp_makes_the_patch_non_uniform(self):
        patch = torch.full((3, 64, 64), 0.5)
        lab = labels((0, 0.5, 0.5, 0.3, 0.3))
        flat = transform(patch, lab, illumination_scale=0.0)
        lit = transform(patch, lab, illumination_scale=0.5)
        x0, y0, x1, y1 = nonzero_bbox(lit[0, 0])
        inner = lit[0, 0, :, y0 + 3 : y1 - 2, x0 + 3 : x1 - 2]
        assert inner.std() > flat[0, 0, :, y0 + 3 : y1 - 2, x0 + 3 : x1 - 2].std()
        assert lit.min() >= 0.0 and lit.max() <= 1.0

    def test_median_pool_can_be_disabled(self):
        patch = torch.rand(3, 64, 64)
        lab = labels((0, 0.5, 0.5, 0.4, 0.4))
        pooled = transform(patch, lab, medianpool_kernel=7)
        raw = transform(patch, lab, medianpool_kernel=0)
        x0, y0, x1, y1 = nonzero_bbox(raw[0, 0])
        box = (slice(None), slice(y0 + 3, y1 - 2), slice(x0 + 3, x1 - 2))
        assert raw[0, 0][box].std() > pooled[0, 0][box].std(), "the median filter should smooth the patch"


class TestConfigValidation:
    """The shipped config must validate, and common mistakes must be caught."""

    @staticmethod
    def base():
        return json.loads(BASE_CFG.read_text())

    def test_base_config_is_valid(self):
        validate_config(EasyDict(self.base()))

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("patch_siz", [64, 64]),  # typo'd key
            ("n_classes", 7),  # disagrees with class_list
            ("loss_target", "obj*conf"),
            ("objective_class_id", 99),
            ("patch_pixel_range", [255, 0]),
            ("patch_size", [64]),
            ("target_size_frac", [0.4, 0.25]),
            ("patch_img_mode", "CMYK"),
            ("patch_alpha", 0),
            ("use_even_odd_images", "some"),
        ],
    )
    def test_invalid_values_are_rejected(self, key, value):
        cfg = self.base()
        cfg[key] = value
        with pytest.raises(ValueError):
            validate_config(EasyDict(cfg))

    def test_missing_key_is_rejected(self):
        cfg = self.base()
        del cfg["start_lr"]
        with pytest.raises(ValueError, match="missing key"):
            validate_config(EasyDict(cfg))


class TestClassIdHandling:
    """objective_class_id accepts null, an int or a list, and all three must reach every consumer."""

    @staticmethod
    def tester():
        pytest.importorskip("pycocotools")
        from test_patch import PatchTester

        return PatchTester

    def test_normalizes_none_int_and_list(self):
        pt = self.tester()
        assert pt.as_class_id_list(None) is None
        assert pt.as_class_id_list(2) == [2]
        assert pt.as_class_id_list([0, 3]) == [0, 3]
        assert pt.as_class_id_list((1, 2)) == [1, 2]

    def test_class_mask_selects_every_listed_id(self):
        pt = self.tester()
        classes = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.0])
        assert pt.class_mask(classes, [1]).tolist() == [False, True, False, False, True]
        assert pt.class_mask(classes, [1, 3]).tolist() == [False, True, False, True, True]

    def test_calc_asr_accepts_a_list_of_target_classes(self):
        pt = self.tester()
        # two gt boxes of class 1 and 2, neither re-detected under the patch
        gt = torch.tensor([[1.0, 10, 10, 60, 60], [2.0, 200, 200, 260, 260]])
        preds = torch.zeros((0, 6))
        for cls_id in (1, [1, 2], None):
            asr = pt.calc_asr(gt, preds, ["a", "b", "c", "d"], cls_id=cls_id)
            assert all(0.0 <= v <= 1.0 for v in asr), f"cls_id={cls_id} gave {asr}"
            assert asr[3] == pytest.approx(1.0, abs=1e-3), f"all boxes hidden should be asr 1.0, cls_id={cls_id}"


class TestBboxPatcher:
    """The detector-side patch augmentation, whose definition a merge with ultralytics dropped."""

    def test_dataloaders_import_chain_is_intact(self):
        """utils.dataloaders is imported by models.common, so a missing BboxPatcher breaks train.py, val.py, detect.py
        and export.py, not just the adv patch code.
        """
        import utils.dataloaders  # noqa: F401
        from utils.dataloaders import BboxPatcher

        assert BboxPatcher is not None

    def test_no_patches_is_a_passthrough(self, tmp_path):
        from adv_patch_gen.utils.bbox_patch_aug import BboxPatcher

        patcher = BboxPatcher(patch_dir=str(tmp_path))
        assert patcher.patches == []
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        boxes = np.array([[0, 8, 8, 40, 40]])
        assert np.array_equal(patcher(img, boxes), img)

    def test_empty_patch_dir_string_does_not_glob_the_cwd(self):
        from adv_patch_gen.utils.bbox_patch_aug import BboxPatcher

        assert BboxPatcher(patch_dir="").patches == []

    def test_patch_is_pasted_onto_the_box(self, tmp_path):
        from adv_patch_gen.utils.bbox_patch_aug import BboxPatcher

        patch_dir = tmp_path / "patches"
        patch_dir.mkdir()
        PILImage.new("RGB", (32, 32), color=(255, 0, 0)).save(patch_dir / "p.png")

        patcher = BboxPatcher(patch_dir=str(patch_dir), patch_apply_prob=1.0, scale_range=(0.3, 0.3))
        assert len(patcher.patches) == 1
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        out = patcher(img, np.array([[0, 32, 32, 96, 96]]))
        assert out.shape == img.shape and out.dtype == img.dtype
        assert not np.array_equal(out, img), "a patch should have been pasted"
        # only the region around the box changes
        assert np.array_equal(out[:24, :24], img[:24, :24]), "corners far from the box must be untouched"

    def test_zero_apply_probability_leaves_the_image_alone(self, tmp_path):
        from adv_patch_gen.utils.bbox_patch_aug import BboxPatcher

        patch_dir = tmp_path / "patches"
        patch_dir.mkdir()
        PILImage.new("RGB", (32, 32), color=(255, 0, 0)).save(patch_dir / "p.png")
        patcher = BboxPatcher(patch_dir=str(patch_dir), patch_apply_prob=0.0)
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        assert np.array_equal(patcher(img, np.array([[0, 32, 32, 96, 96]])), img)

    def test_degenerate_boxes_are_skipped(self, tmp_path):
        from adv_patch_gen.utils.bbox_patch_aug import BboxPatcher

        patch_dir = tmp_path / "patches"
        patch_dir.mkdir()
        PILImage.new("RGB", (32, 32), color=(255, 0, 0)).save(patch_dir / "p.png")
        patcher = BboxPatcher(patch_dir=str(patch_dir), patch_apply_prob=1.0)
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        boxes = np.array([[0, 50, 50, 50, 50], [0, 60, 60, 40, 40]])  # zero and negative extent
        assert np.array_equal(patcher(img, boxes), img)


class TestDataset:
    """Image/label pairing must name what is wrong rather than fail an opaque assert."""

    def test_missing_label_file_names_itself(self, tmp_path):
        img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
        img_dir.mkdir()
        lab_dir.mkdir()
        for name in ("a", "b"):
            PILImage.new("RGB", (32, 32)).save(img_dir / f"{name}.jpg")
        (lab_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # b.txt deliberately absent

        with pytest.raises(FileNotFoundError, match="b.txt"):
            YOLODataset(str(img_dir), str(lab_dir), max_labels=4, model_in_sz=(64, 64))

    def test_labels_are_padded_and_area_sorted(self, tmp_path):
        img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
        img_dir.mkdir()
        lab_dir.mkdir()
        PILImage.new("RGB", (64, 64)).save(img_dir / "a.jpg")
        (lab_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.4 0.4\n")

        img, lab = YOLODataset(str(img_dir), str(lab_dir), max_labels=5, model_in_sz=(64, 64))[0]
        assert img.shape == (3, 64, 64)
        assert lab.shape == (5, 5)
        assert lab[0][0] == 1, "labels should be sorted by descending box area"
        assert lab[2:].abs().sum() == 0, "unused label rows should be zero padding"
