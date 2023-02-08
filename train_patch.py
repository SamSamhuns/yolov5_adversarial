"""
Training code for Adversarial patch training

python train_patch.py --cfg config_json_file
"""
import os
import os.path as osp
import time
import json
from contextlib import nullcontext

import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
from torch import optim, autograd
from torch.cuda.amp import autocast
from torchvision import transforms

from tensorboardX import SummaryWriter
from tensorboard import program

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

from adv_patch_gen.utils.config_parser import get_argparser, load_config_object
from adv_patch_gen.utils.common import is_port_in_use
from adv_patch_gen.utils.dataset import YOLODataset
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from adv_patch_gen.utils.loss import MaxProbExtractor, SaliencyLoss, TotalVariationLoss, NPSLoss

# optionally set seed for repeatability
SEED = None
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
# setting benchmark to False reduces training time for our setup
torch.backends.cudnn.benchmark = False


class PatchTrainer:
    """
    Module for training on dataset to generate adv patches
    """

    def __init__(self, cfg: edict):
        self.cfg = cfg
        self.dev = select_device(cfg.device)

        model = DetectMultiBackend(cfg.weights_file, device=self.dev, dnn=False, data=None, fp16=False)
        self.model = model.eval()

        self.patch_transformer = PatchTransformer(cfg.target_size_frac, self.dev).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)
        self.prob_extractor = MaxProbExtractor(cfg).to(self.dev)
        self.sal_loss = SaliencyLoss().to(self.dev)
        self.nps_loss = NPSLoss(cfg.triplet_printfile, cfg.patch_size).to(self.dev)
        self.tv_loss = TotalVariationLoss().to(self.dev)

        # freeze entire detection model
        for param in self.model.parameters():
            param.requires_grad = False

        # set log dir
        cfg.log_dir = osp.join(cfg.log_dir, f'{time.strftime("%Y%m%d-%H%M%S")}_{cfg.patch_name}')
        self.writer = self.init_tensorboard(cfg.log_dir, cfg.tensorboard_port)
        # save config parameters to tensorboard logs
        for cfg_key, cfg_val in cfg.items():
            self.writer.add_text(cfg_key, str(cfg_val))

        # load dataset
        self.train_loader = torch.utils.data.DataLoader(
            YOLODataset(cfg.image_dir,
                        cfg.label_dir,
                        cfg.max_labels,
                        cfg.model_in_sz,
                        cfg.use_even_odd_images,
                        shuffle=True),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True if self.dev.type == "cuda" else False)
        self.epoch_length = len(self.train_loader)

    def init_tensorboard(self, log_dir: str = None, port: int = 6006, run_tb=True):
        """
        Initialize tensorboard with optional name
        """
        if run_tb:
            while is_port_in_use(port) and port < 65535:
                port += 1
                print(f"Port {port} is currently in use. Switching to {port} for tensorboard logging")

            tboard = program.TensorBoard()
            tboard.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
            url = tboard.launch()
            print(f"Tensorboard logger started on {url}")

        if log_dir:
            return SummaryWriter(log_dir)
        return SummaryWriter()

    def train(self) -> None:
        """
        Optimize a patch to generate an adversarial example.
        """

        # make output dirs
        patch_dir = osp.join(self.cfg.log_dir, "patches")
        os.makedirs(patch_dir, exist_ok=True)
        log_file = osp.join(self.cfg.log_dir, 'log.txt')
        if self.cfg.debug_mode:
            os.makedirs(osp.join(self.cfg.log_dir, "patch_applied_imgs"), exist_ok=True)
        # dump cfg json file
        with open(osp.join(self.cfg.log_dir, "cfg.json"), 'w', encoding='utf-8') as json_f:
            json.dump(self.cfg, json_f, ensure_ascii=False, indent=4)

        # fix loss targets
        loss_target = self.cfg.loss_target
        if loss_target == "obj":
            self.cfg.loss_target = lambda obj, cls: obj
        elif loss_target == "cls":
            self.cfg.loss_target = lambda obj, cls: cls
        elif loss_target == "obj * cls":
            self.cfg.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(
                f"Loss target {loss_target} not been implemented")

        # Generate init patch
        if self.cfg.patch_src == 'gray':
            adv_patch_cpu = self.generate_patch("gray")
        elif self.cfg.patch_src == 'random':
            adv_patch_cpu = self.generate_patch("random")
        else:
            adv_patch_cpu = self.read_image(self.cfg.patch_src)
        adv_patch_cpu.requires_grad = True

        optimizer = optim.Adam(
            [adv_patch_cpu], lr=self.cfg.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=50)

        for epoch in range(self.cfg.n_epochs):
            print(epoch)
            out_patch_path = osp.join(
                patch_dir, f"{self.cfg.patch_name}_epoch_{epoch}.jpg")
            ep_loss = 0
            min_tv_loss = torch.tensor(self.cfg.min_tv_loss).to(self.dev)

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(self.train_loader),
                                                        desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.set_detect_anomaly(mode=True if self.cfg.debug_mode else False):
                    img_batch = img_batch.to(self.dev)
                    lab_batch = lab_batch.to(self.dev)
                    adv_patch = adv_patch_cpu.to(self.dev)
                    adv_batch_t = self.patch_transformer(
                        adv_patch, lab_batch, self.cfg.model_in_sz,
                        do_transforms=self.cfg.transform_patches,
                        do_rotate=self.cfg.rotate_patches, rand_loc=False)
                    p_img_batch = self.patch_applier(
                        img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(
                        p_img_batch, (self.cfg.model_in_sz[0], self.cfg.model_in_sz[1]))

                    if self.cfg.debug_mode:
                        img = p_img_batch[1, :, :, ]
                        img = transforms.ToPILImage()(img.detach().cpu())
                        img.save(osp.join(self.cfg.log_dir, "patch_applied_imgs", f"b_{i_batch}.jpg"))

                    with autocast() if self.cfg.use_amp else nullcontext():
                        output = self.model(p_img_batch)[0]
                        max_prob = self.prob_extractor(output)
                        nps = self.nps_loss(adv_patch) if self.cfg.nps_mult != 0 else torch.tensor([0], device=self.dev)
                        tv = self.tv_loss(adv_patch) if self.cfg.tv_mult != 0 else torch.tensor([0], device=self.dev)

                    nps_loss = nps * self.cfg.nps_mult
                    tv_loss = torch.max(tv * self.cfg.tv_mult, min_tv_loss)
                    det_loss = torch.mean(max_prob)

                    loss = det_loss + nps_loss + tv_loss
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # keep patch in image range
                    adv_patch_cpu.data.clamp_(0, 1)

                    if i_batch % 15 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar(
                            'total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'misc/epoch', epoch, iteration)
                        self.writer.add_scalar(
                            'misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image(
                            'patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(self.train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        # torch.cuda.empty_cache()  # note emptying cache adds too much overhead
            ep_loss = ep_loss/len(self.train_loader)
            scheduler.step(ep_loss)

            # save patch after every 1 epoch(s)
            if epoch % 1 == 0:
                img = transforms.ToPILImage('RGB')(adv_patch_cpu)
                img.save(out_patch_path)
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                # torch.cuda.empty_cache()  # note emptying cache adds too much overhead

    def generate_patch(self, patch_type: str) -> torch.Tensor:
        """
        Generate a random patch as a starting point for optimization.

        Arguments:
            patch_type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        """
        p_w, p_h = self.cfg.patch_size
        if patch_type == 'gray':
            adv_patch_cpu = torch.full((3, p_h, p_w), 0.5)
        elif patch_type == 'random':
            adv_patch_cpu = torch.rand((3, p_h, p_w))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        transforms_resize = transforms.Resize(self.cfg.patch_size)
        patch_img = transforms_resize(patch_img)
        adv_patch_cpu = transforms.ToTensor()(patch_img)
        return adv_patch_cpu


def main():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = load_config_object(args.config)
    trainer = PatchTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
