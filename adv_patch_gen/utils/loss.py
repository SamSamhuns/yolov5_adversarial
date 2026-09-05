"""Loss functions used in patch generation."""

import torch
from torch import nn


class MaxProbExtractor(nn.Module):
    """Extracts the detection confidence the patch is optimizing against, from a YOLO head's raw output.

    Handles both head layouts: yolov5 [batch, n_preds, 5 + n_classes] xywh, objectness, per class scores (all sigmoid
    activated) yolov8+ [batch, 4 + n_classes, n_preds] xywh, per class scores, transposed and with no objectness

    config keys used:
        n_classes: number of classes the detector predicts
        objective_class_id: None for an untargeted attack, else an int or list of class ids to suppress
        loss_target: callable(objectness, class_conf) chosen by the loss_target config string
        loss_topk: mean of the k highest scoring predictions per image. 1 reproduces the original max.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        cls_id = config.objective_class_id
        self.objective_class_ids = None if cls_id is None else ([cls_id] if isinstance(cls_id, int) else list(cls_id))
        self.topk = max(1, int(getattr(config, "loss_topk", 1) or 1))
        self._warned_no_objectness = False

    def _split_head(self, output: torch.Tensor):
        """Return (objectness, class_confs) as [batch, n_preds] and [batch, n_preds, n_classes]."""
        n_cls = self.config.n_classes
        if output.size(-1) == 5 + n_cls:  # yolov5
            return output[..., 4], output[..., 5 : 5 + n_cls]
        if output.size(1) == 4 + n_cls:  # yolov8/v11, transposed and without an objectness channel
            preds = output.transpose(1, 2)  # [batch, n_preds, 4 + n_classes]
            class_confs = preds[..., 4 : 4 + n_cls]
            if not self._warned_no_objectness:
                print(
                    "NOTE: detector head has no objectness channel, objectness is treated as 1. "
                    'Use loss_target "cls" or "obj * cls", "obj" alone carries no gradient for this head.'
                )
                self._warned_no_objectness = True
            return torch.ones_like(class_confs[..., 0]), class_confs
        raise ValueError(
            f"Unrecognized detector output {tuple(output.shape)} for n_classes={n_cls}. "
            f"Expected [batch, n_preds, {5 + n_cls}] (yolov5) or [batch, {4 + n_cls}, n_preds] (yolov8+)."
        )

    def forward(self, output: torch.Tensor):
        """Output is the raw head tensor, see _split_head for the layouts accepted."""
        objectness_score, class_confs = self._split_head(output)

        if self.objective_class_ids is not None:
            # head output is already sigmoid activated, so class_confs are per class probs in [0, 1].
            # take the best of the targeted classes for each prediction
            class_confs = class_confs[..., self.objective_class_ids].max(dim=2)[0]
        else:
            # get class with highest conf for each box if objective_class_id is None
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, n_preds, n_classes] -> [batch, n_preds]

        confs_if_object = self.config.loss_target(objectness_score, class_confs)
        if self.topk == 1:
            return torch.max(confs_if_object, dim=1)[0]
        # mean of the topk highest scoring predictions, so more than one prediction per image
        # gets gradient. A single max sends gradient to 1 of ~25k predictions at 640x640.
        k = min(self.topk, confs_if_object.size(1))
        return torch.topk(confs_if_object, k, dim=1)[0].mean(dim=1)


class SaliencyLoss(nn.Module):
    """Implementation of the colorfulness metric as the saliency loss.

    The smaller the value, the less colorful the image. The metric is already scale invariant (it is a statistic over
    the opponent color channels), so it is not normalized by patch size. Reference:
    https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Float Tensor of shape [C, H, W] where C=3 (R, G, B channels).
        """
        assert adv_patch.shape[0] == 3
        r, g, b = adv_patch
        rg = r - g
        yb = 0.5 * (r + g) - b

        mu_rg, sigma_rg = torch.mean(rg) + 1e-8, torch.std(rg) + 1e-8
        mu_yb, sigma_yb = torch.mean(yb) + 1e-8, torch.std(yb) + 1e-8
        return torch.sqrt(sigma_rg**2 + sigma_yb**2) + (0.3 * torch.sqrt(mu_rg**2 + mu_yb**2))


class TotalVariationLoss(nn.Module):
    """TotalVariationLoss: calculates the total variation of a patch. Module providing the functionality necessary to
    calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W].
        """
        # calc diff in patch rows
        tvcomp_r = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0)
        # calc diff in patch columns
        tvcomp_c = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        tv = tvcomp_r + tvcomp_c
        return tv / torch.numel(adv_patch)


class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch. Module providing the functionality necessary to
    calculate the non-printability score (NMS) of an adversarial patch. However, a summation of the differences is
    used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf.

    Args:
        triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines.
    """

    def __init__(self, triplet_scores_fpath: str):
        super().__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(triplet_scores_fpath), requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidean distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = adv_patch - self.printability_array + 0.000001
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    @staticmethod
    def get_printability_array(triplet_scores_fpath: str) -> torch.Tensor:
        """Get printability tensor of shape [n_triplets, 3, 1, 1] holding the rgb triplets (range [0,1]) loaded from
        triplet_scores_fpath. It broadcasts against a [3, H, W] patch of any size.

        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines.
        """
        ref_triplet_list = []
        with open(triplet_scores_fpath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ref_triplet_list.append([float(v) for v in line.strip().split(",")])
        if not ref_triplet_list:
            raise ValueError(f"No rgb triplets found in {triplet_scores_fpath}")
        return torch.tensor(ref_triplet_list, dtype=torch.float32).view(len(ref_triplet_list), 3, 1, 1)
