import torch
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler

from data.augment.build import build_transform

from utils.configurable import configurable
from utils.registry import Registry 
from utils.dist import get_world_size, get_rank

DATASET_REGISTRY = Registry("Datasets")


def build_dataset(cfg):
    transform = build_transform(cfg)
    dataset = DATASET_REGISTRY.get(cfg.data.dataset.name)(cfg=cfg, transform=transform)
    train_data, val_data = dataset.get_data()
    return train_data, val_data

def _cfg_to_trainloader(cfg):
    return {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,
        "drop_last": cfg.data.drop_last,

        "distributed": cfg.distributed,
        "world_size": get_world_size(),
        "rank": get_rank(),
    }

@configurable(from_config=_cfg_to_trainloader)
def build_train_dataloader(
    train_dataset,
    *,
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool, 
    drop_last: bool,
    distributed: bool,
    world_size: int,
    rank: int,
):
    if distributed:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        sampler = RandomSampler(train_dataset)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return train_loader


def _cfg_to_valloader(cfg):
    return {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,

        "distributed": cfg.distributed,
        "distributed_eval": cfg.dist_eval,
        "world_size": get_world_size(),
        "rank": get_rank(),
    }

@configurable(from_config=_cfg_to_valloader)
def build_val_dataloader(
    val_dataset,
    *,
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool, 
    distributed: bool,
    distributed_eval: bool,
    world_size: int,
    rank: int,
):
    if distributed and distributed_eval:
        if len(val_dataset) % world_size != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return val_loader
