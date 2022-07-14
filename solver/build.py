import torch.optim as optim

from utils.registry import Registry

OPTIMIZER_REGISTRY = Registry("Optimizer")
LR_SCHEDULER_REGISTRY = Registry("LRscheduler")

def build_base_optimizer(cfg, parameters):
    base_optimizer_type = cfg.optimizer.base_optim
    base_optimizer_kwargs = cfg.optimizer[base_optimizer_type]
    opt_kwargs = dict(
        params=parameters,
        **base_optimizer_kwargs,
    )
    base_optimizer = getattr(optim, base_optimizer_type)(**opt_kwargs)
    return base_optimizer


def build_optimizer(cfg, model):
    base_optimizer = build_base_optimizer(cfg, model.parameters())
    if cfg.optimizer.name != cfg.optimizer.base_optim:
        optimizer = OPTIMIZER_REGISTRY.get(cfg.optimizer.name)(params=model.parameters(), base_optimizer=base_optimizer, cfg=cfg)
    else:
        optimizer = base_optimizer
    return optimizer, base_optimizer


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(cfg.lr_scheduler.name)(optimizer=optimizer, cfg=cfg)
    return lr_scheduler
