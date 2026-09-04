"""Modules for creating adversarial object patch."""

import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_patch_gen.utils.median_pool import MedianPool2d


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Scales and places one patch per label box and
    returns them on image sized canvases, zero everywhere a patch does not cover.

    grid_sample samples the patch itself rather than a copy padded out to the model input size, which keeps the
    intermediates at patch resolution instead of allocating two [bsize * max_labels, c, H, W] tensors.
    """

    def __init__(
        self,
        t_size_frac: Union[float, Tuple[float, float]] = 0.3,
        mul_gau_mean: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mul_gau_std: Union[float, Tuple[float, float]] = 0.1,
        x_off_loc: Tuple[float, float] = [-0.25, 0.25],
        y_off_loc: Tuple[float, float] = [-0.25, 0.25],
        dev: torch.device = torch.device("cuda:0"),
        medianpool_kernel: int = 7,
        perspective_scale: float = 0.0,
        illumination_scale: float = 0.0,
    ):
        super(PatchTransformer, self).__init__()
        # convert to duplicated lists/tuples to unpack and send to np.random.uniform
        self.t_size_frac = [t_size_frac, t_size_frac] if isinstance(t_size_frac, float) else t_size_frac
        self.m_gau_mean = [mul_gau_mean, mul_gau_mean] if isinstance(mul_gau_mean, float) else mul_gau_mean
        self.m_gau_std = [mul_gau_std, mul_gau_std] if isinstance(mul_gau_std, float) else mul_gau_std
        assert len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2, (
            "Range must have 2 values"
        )
        self.x_off_loc = x_off_loc
        self.y_off_loc = y_off_loc
        self.dev = dev
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        # a median filter over the patch each forward. Larger kernels low pass the patch harder and
        # route gradient to fewer pixels, 0 or 1 disables it entirely.
        self.medianpooler = MedianPool2d(kernel_size=medianpool_kernel, same=True) if medianpool_kernel > 1 else None
        # out of plane tilt, so the patch is not always seen fronto-parallel
        self.perspective_scale = perspective_scale
        # linear brightness ramp across the patch, standing in for directional light and soft shadow
        self.illumination_scale = illumination_scale

        self.tensor = torch.FloatTensor if "cpu" in str(dev) else torch.cuda.FloatTensor

    def forward(
        self,
        adv_patch,
        lab_batch,
        model_in_sz,
        use_mul_add_gau=True,
        do_transforms=True,
        do_rotate=True,
        rand_loc=True,
        do_perspective=True,
    ):
        # add gaussian noise to reduce contrast with a stohastic process
        p_c, p_h, p_w = adv_patch.shape
        if use_mul_add_gau:
            mul_gau = torch.normal(
                np.random.uniform(*self.m_gau_mean),
                np.random.uniform(*self.m_gau_std),
                (p_c, p_h, p_w),
                device=self.dev,
            )
            add_gau = torch.normal(0, 0.001, (p_c, p_h, p_w), device=self.dev)
            adv_patch = adv_patch * mul_gau + add_gau
        adv_patch = adv_patch.unsqueeze(0)
        if self.medianpooler is not None:
            adv_patch = self.medianpooler(adv_patch)
        m_h, m_w = model_in_sz
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1
        )  # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms
        if do_transforms:
            # Create random contrast tensor
            contrast = self.tensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random brightness tensor
            brightness = self.tensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random noise tensor
            noise = self.tensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            adv_batch = adv_batch * contrast + brightness + noise

            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Linear illumination ramp across the patch, one random direction per patch, standing in for
        # directional lighting and soft shadow. Applied at patch resolution, before the warp.
        if self.illumination_scale > 0:
            p_h_cur, p_w_cur = adv_batch.size(-2), adv_batch.size(-1)
            ramp_y = torch.linspace(-1, 1, p_h_cur, device=adv_batch.device).view(1, 1, 1, p_h_cur, 1)
            ramp_x = torch.linspace(-1, 1, p_w_cur, device=adv_batch.device).view(1, 1, 1, 1, p_w_cur)
            dir_y = self.tensor(batch_size).uniform_(-1, 1)[:, :, None, None, None]
            dir_x = self.tensor(batch_size).uniform_(-1, 1)[:, :, None, None, None]
            illum = 1.0 + self.illumination_scale * (dir_x * ramp_x + dir_y * ramp_y)
            adv_batch = torch.clamp(adv_batch * illum, 0.000001, 0.99999)

        # lab_batch is zero-padded up to max_labels, so drop patches for the zero area filler rows.
        # The mask also marks where a patch landed, which PatchApplier needs since the clamp below
        # lifts every pixel off exact zero.
        valid_box = (lab_batch[..., 3] * lab_batch[..., 4]) > 0  # [bsize, max_bbox_labels]
        # one channel is enough, the mask is identical across channels and broadcasts on the multiply
        msk_batch = valid_box[:, :, None, None, None].to(adv_batch.dtype)
        msk_batch = msk_batch.expand(-1, -1, 1, adv_batch.size(-2), adv_batch.size(-1))

        # Rotation and rescaling transforms
        anglesize = lab_batch.size(0) * lab_batch.size(1)
        if do_rotate:
            angle = self.tensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = self.tensor(anglesize).fill_(0)

        # Resizes and rotates
        p_h_pooled, p_w_pooled = adv_batch.size(-2), adv_batch.size(-1)
        tsize = np.random.uniform(*self.t_size_frac)
        # patch is sized off the bbox diagonal in model input pixels
        target_size = tsize * torch.sqrt(((lab_batch[:, :, 3] * m_w) ** 2) + ((lab_batch[:, :, 4] * m_h) ** 2))
        # zero area filler rows would give scale 0 and hence inf in theta below, so give them a
        # full canvas scale instead. Their patches are zeroed out by msk_batch after grid_sample.
        target_size = torch.where(valid_box, target_size, torch.full_like(target_size, float(m_w)))

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if rand_loc:
            off_x = targetoff_x * (self.tensor(targetoff_x.size()).uniform_(*self.x_off_loc))
            target_x = target_x + off_x
            off_y = targetoff_y * (self.tensor(targetoff_y.size()).uniform_(*self.y_off_loc))
            target_y = target_y + off_y
        # grid_sample reads the patch directly rather than a canvas sized copy of it, so the scale
        # is the rendered fraction of each image axis. Normalizing x and y separately also keeps
        # the patch square for a non-square model_in_sz, and keeps its aspect for a non-square patch.
        scale_x = (target_size / m_w).view(anglesize)
        scale_y = (target_size * (p_h_pooled / p_w_pooled) / m_h).view(anglesize)

        # reshape not view: both are expanded views when no transforms materialized them
        s = adv_batch.size()
        adv_batch = adv_batch.reshape(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.reshape(s[0] * s[1], 1, s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation/rescale matrix
        # Theta = input batch of affine matrices with shape (N×2×3) for 2D or (N×3×4) for 3D
        theta = self.tensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale_x
        theta[:, 0, 1] = sin / scale_x
        theta[:, 0, 2] = (tx * cos + ty * sin) / scale_x
        theta[:, 1, 0] = -sin / scale_y
        theta[:, 1, 1] = cos / scale_y
        theta[:, 1, 2] = (-tx * sin + ty * cos) / scale_y

        out_shape = (s[0] * s[1], s[2], m_h, m_w)
        # The sampling grid comes only from label boxes and random draws, none of which carry
        # gradient, so it can be built under no_grad and perspective divided in place.
        with torch.no_grad():
            grid = F.affine_grid(theta, out_shape, align_corners=False)
            if do_perspective and self.perspective_scale > 0:
                # Divide the grid by a plane w = 1 + px*x + py*y. That is the inverse map of a patch
                # lying on a surface tilted out of the image plane, i.e. the usual keystone effect.
                px = self.tensor(anglesize).uniform_(-self.perspective_scale, self.perspective_scale)
                py = self.tensor(anglesize).uniform_(-self.perspective_scale, self.perspective_scale)
                xs = torch.linspace(-1, 1, m_w, device=grid.device).view(1, 1, m_w)
                ys = torch.linspace(-1, 1, m_h, device=grid.device).view(1, m_h, 1)
                w_plane = 1.0 + px[:, None, None] * xs + py[:, None, None] * ys
                # clamp keeps the plane from crossing zero, which would mirror or blow up the patch
                grid.div_(w_plane.clamp(min=0.2).unsqueeze(-1))
        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
        msk_batch_t = F.grid_sample(msk_batch, grid, align_corners=False)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], m_h, m_w)
        msk_batch_t = msk_batch_t.view(s[0], s[1], 1, m_h, m_w)

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    The patch (adv_batch) has the same size as the image, just is zero everywhere there isn't a patch.
    If patch_alpha == 1 (default), just overwrite the background image values with the patch values.
    Else, blend the patch with the image
    See: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
         https://stackoverflow.com/questions/49737541/merge-two-images-with-alpha-channel/49738078
        I = \alpha F + (1 - \alpha) B
            F = foregraound (patch, or adv_batch)
            B = background (image, or img_batch)
    """

    def __init__(self, patch_alpha: float = 1):
        super(PatchApplier, self).__init__()
        self.patch_alpha = patch_alpha

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # replace image values with patch values
            if self.patch_alpha == 1:
                img_batch = torch.where((adv == 0), img_batch, adv)
            # alpha blend
            else:
                # get combo of image and adv
                alpha_blend = self.patch_alpha * adv + (1.0 - self.patch_alpha) * img_batch
                # apply alpha blend where the patch is non-zero
                img_batch = torch.where((adv == 0), img_batch, alpha_blend)

        return img_batch
