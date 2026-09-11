# Adversarial Patch Training Config Reference

```
    "image_dir": "data/train/images",
    "label_dir": "data/train/labels",
    "val_image_dir": "data/val/images",     # epoch freq for running validation run. 1 means validate after every epoch. 0 or null means no val
    "use_even_odd_images": "all",           # (str), ('all', 'even', 'odd'): use images with even/odd numbers in the last char of their filenames
    "log_dir": "runs/train_adversarial",
    "run_tensorboard": true,                # bool: also serve tensorboard on tensorboard_port. Set false for headless/docker runs. Defaults to true when omitted
    "tensorboard_port": 8994,
    "tensorboard_batch_log_interval": 15,
    "weights_file": "runs/weights/best.pt",
    "model_backend": "yolov5",              # str: "yolov5" for this repo's DetectMultiBackend, "ultralytics" for v8/v11 weights.
                                            # Only torch backends can be TRAINED against, exports sever the autograd graph. Defaults to yolov5
    "triplet_printfile": "triplets.csv",
    "device": "cuda:0",                     # (str): 'cpu' or 'cuda' or 'cuda:0,1,2,3'
    "seed": 42,                             # int: RNG seed for repeatability. Use null for a non deterministic run. Defaults to 42 when omitted
    "use_amp": true,
    "patch_name": "base",
    "val_epoch_freq": 10,                   # int: epochs between validation runs. best.png can only be picked from epochs that ran
                                            # val, so a coarse value makes "best" close to arbitrary
    "val_target_size_frac": null,           # float: patch scale used for validation. null uses the midpoint of target_size_frac
    "val_max_images": 200,                  # int: cap on val images scored per epoch, a fixed prefix of the sorted list.
                                            # null scores the whole val dir. Lets val_epoch_freq stay small
    "val_seed": 0,                          # int: pins the val nuisance draws so ASR is comparable epoch to epoch. The training
                                            # RNG stream is saved and restored around val, so this does not perturb training
    "patch_save_epoch_freq": 1,             # int freq for saving patches. 1 means save after every epoch
    "model_in_sz": [640, 640],              # (int, int): model input height, width
    "patch_src": "gray",                    # str: gray random, or path_to_init_patch
    "patch_img_mode": "RGB",                # str: patch channel image mode. Currently RGB * L supported
    "patch_size": [64, 64],                 # (int, int): must be (height, width)
    "objective_class_id": null,             # int or [int, ...]: class id(s) to target. Use null for a general attack on all classes
    "min_pixel_area": null,                 # int: min pixel area to use for training. Pixel area chosen after resizing to model in size
    "augment_image": true,                  # bool: photometric augmentation (blur, color jitter, sharpness) plus label aware hflip
    "augment_stage": "post",                # str: "post" augments the PATCHED image so the patch is degraded by the same camera
                                            # response as the scene (physically correct). "pre" is the old behavior. Defaults to post
    "target_size_frac": 0.3,                # float: patch proportion size compared to bbox size. Range also accepted i.e. [0.25, 0.4]
    "use_mul_add_gau": true,                # bool: use mul & add gaussian noise or not to patches
    "mul_gau_mean": 0.5,                    # float: mul gaussian noise mean (reduces contrast) mean. Range also accepted i.e. [0.25, 0.4]
    "mul_gau_std": 0.1,                     # float: mul gaussian noise std (Adds rand noise)
    "random_patch_loc": true,               # bool: place/translate patches randomly on bbox
    "x_off_loc": [-0.25, 0.25],             # [float, float]: left, right x-axis disp from bbox center
    "y_off_loc": [-0.25, 0.25],             # [float, float]: top, bottom y-axis disp from bbox center
    "rotate_patches": true,                 # bool: rotate patches or not
    "perspective_scale": 0.1,               # float [0,1): out of plane tilt of the patch (keystone). 0 disables
    "illumination_scale": 0.2,              # float [0,1): linear brightness ramp across the patch, directional light/shadow. 0 disables
    "blur_sigma_max": 0.6,                  # float [0,10): defocus blur, one sigma drawn per rendered patch, standing in for objects
                                            # at different distances from the camera. 0 disables
    "transform_patches": true,              # bool: add bightness, contrast and noise transforms to patches or not
    "patch_pixel_range": [0, 255],          # [int, int]: patch pixel range, range is [0, 255], numbers div by 255 in patches
    "patch_alpha": 1,                       # float: patch opacity, recommended to set to 1
    "class_list": ["class1", "class2"],
    "n_classes": 2,
    "n_epochs": 300,
    "max_labels": 48,
    "start_lr": 0.03,
    "min_tv_loss": 0.1,
    "sal_mult": 0.1,                        # float: saliency (colorfulness) loss weight. The loss is an absolute metric in ~[0, 1],
                                            # i.e. on the same scale as the det loss, and it suppresses exactly the colorfulness that
                                            # makes a patch strong, so keep it well below 1
    "tv_mult": 2.5,
    "nps_mult": 0.01,                       # float: Use 0.01 when not using sal. With sal use 0.001
    "batch_size": 8,
    "debug_mode": false,                    # bool: if yes, images with adv drawn saved during each batch
    "loss_target": "obj * cls",             # str: 'obj', 'cls', 'obj * cls'. A v8/v11 head has no objectness, use 'cls' or 'obj * cls' there
    "loss_topk": 10,                        # int >=1: mean of the k highest scoring predictions. 1 is the original single max, which
                                            # sends gradient to 1 of ~25k predictions per image at 640x640
    "medianpool_kernel": 3,                 # odd int: median filter over the patch each forward. Larger low passes the patch harder
                                            # and routes gradient to fewer pixels. 7 on a 64px patch is 11% of its width. 0 or 1 disables
    "printable_quantize": false             # bool: snap patch pixels to the triplet_printfile palette each forward, with a straight
                                            # through gradient, so the optimized patch is exactly printable. Stronger and coarser than
                                            # nps_mult; A/B the two rather than leaning hard on both
```
