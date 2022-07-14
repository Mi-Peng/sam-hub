import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

import contextlib

from utils.dist import init_distributed_model


class Freeze_BN_Running_Stats:
    def __init__(self, model, single_bn_stats: bool) -> None:
        self.model = model
        self.single_bn_stats = single_bn_stats

    def __enter__(self):
        def _disable_running_stats(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0
        if self.single_bn_stats:
            self.model.apply(_disable_running_stats)
        else:
            return
    def __exit__(self, exc_type, exc_val, exc_tb):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum
        if self.single_bn_stats:
            self.model.apply(_enable)
        else:
            return


def Sync_Perturbation(model, sync_perturbation):
    if torch.distributed.is_initialized() and not sync_perturbation:
        return model.no_sync()
    else:
        return contextlib.ExitStack()