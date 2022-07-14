import math
import torchvision.transforms as transforms
from data.augment.cutout import Cutout



from utils.configurable import configurable


def _cfg_to_transform(cfg):
    return {
        "input_size": cfg.data.dataset.input_size,
        "scale": cfg.data.dataaug.scale or (0.08, 1.0),
        "ratio": cfg.data.dataaug.ratio or (3./4., 4./3.),
        "crop_pct": cfg.data.dataaug.crop_pct or 0.875,
        "mean": cfg.data.dataset.mean,
        "std": cfg.data.dataset.std,
        "p_hflip": cfg.data.dataaug.p_hflip or 0.5,
        "cutout_p": cfg.data.dataaug.cutout.p or 0.0,
        "cutout_size": cfg.data.dataaug.cutout.size or cfg.data.dataset.input_size // 2
    }


@configurable(from_config=_cfg_to_transform)
def build_transform(
    input_size,
    scale,
    ratio,
    crop_pct,
    mean, std,
    p_hflip,
    cutout_p,
    cutout_size,
):
    # for model training
    train_transform_list = [
        transforms.RandomResizedCrop(
            size = input_size,
            scale = scale,
            ratio = ratio,
        ),
    ]
    if p_hflip > 0: train_transform_list.append(transforms.RandomHorizontalFlip(p_hflip))
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if cutout_p > 0:
        train_transform_list.append(Cutout(cutout_size, cutout_p))
    train_transform = transforms.Compose(train_transform_list)


    # for model evaluating
    scale_size = int(math.floor(input_size / crop_pct))
    val_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform