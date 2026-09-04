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
    "triplet_printfile": "triplets.csv",
    "device": "cuda:0",                     # (str): 'cpu' or 'cuda' or 'cuda:0,1,2,3'
    "seed": 42,                             # int: RNG seed for repeatability. Use null for a non deterministic run. Defaults to 42 when omitted
    "use_amp": true,
    "patch_name": "base",
    "val_epoch_freq": 100,
    "patch_save_epoch_freq": 1,             # int freq for saving patches. 1 means save after every epoch
    "model_in_sz": [640, 640],              # (int, int): model input height, width
    "patch_src": "gray",                    # str: gray random, or path_to_init_patch
    "patch_img_mode": "RGB",                # str: patch channel image mode. Currently RGB * L supported
    "patch_size": [64, 64],                 # (int, int): must be (height, width)
    "objective_class_id": null,             # int: class id to target for adv attack. Use null for general attack for all classes
    "min_pixel_area": null,                 # int: min pixel area to use for training. Pixel area chosen after resizing to model in size
    "target_size_frac": 0.3,                # float: patch proportion size compared to bbox size. Range also accepted i.e. [0.25, 0.4]
    "use_mul_add_gau": true,                # bool: use mul & add gaussian noise or not to patches
    "mul_gau_mean": 0.5,                    # float: mul gaussian noise mean (reduces contrast) mean. Range also accepted i.e. [0.25, 0.4]
    "mul_gau_std": 0.1,                     # float: mul gaussian noise std (Adds rand noise)
    "random_patch_loc": true,               # bool: place/translate patches randomly on bbox
    "x_off_loc": [-0.25, 0.25],             # [float, float]: left, right x-axis disp from bbox center
    "y_off_loc": [-0.25, 0.25],             # [float, float]: top, bottom y-axis disp from bbox center
    "rotate_patches": true,                 # bool: rotate patches or not
    "transform_patches": true,              # bool: add bightness, contrast and noise transforms to patches or not
    "patch_pixel_range": [0, 255],          # [int, int]: patch pixel range, range is [0, 255], numbers div by 255 in patches
    "patch_alpha": 1,                       # float: patch opacity, recommended to set to 1
    "class_list": ["class1", "class2"],
    "n_classes": 2,
    "n_epochs": 300,
    "max_labels": 48,
    "start_lr": 0.03,
    "min_tv_loss": 0.1,
    "sal_mult": 1.0,                        # float: saliency (colorfulness) loss weight. The loss is an absolute metric in ~[0, 1],
                                            # i.e. on the same scale as the det loss, so retune this if your config predates that fix
    "tv_mult": 2.5,
    "nps_mult": 0.01,                       # float: Use 0.01 when not using sal. With sal use 0.001
    "batch_size": 8,
    "debug_mode": false,                    # bool: if yes, images with adv drawn saved during each batch
    "loss_target": "obj * cls"              # str: 'obj', 'cls', 'obj * cls'
```
