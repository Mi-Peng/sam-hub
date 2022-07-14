import os
import functools
import torch
import torch.distributed as dist

from omegaconf import open_dict

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process(func=None):
    if func is not None:  # used as decorator
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if get_rank() == 0:
                return func(*args, **kwargs)
        return wrapped
    else:  # used as a function
        return get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def init_distributed_model(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        with open_dict(cfg):
            cfg.rank = int(os.environ["RANK"])
            cfg.world_size = int(os.environ["WORLD_SIZE"])
            cfg.gpu = int(os.environ["LOCAL_RANK"])
            cfg.distributed = True
        print('Using distributed mode: WORLD_SIZE:{}.'.format(cfg.world_size))
    else:
        print('Not using distributed mode.')
        with open_dict(cfg):
            cfg.distributed = False
        return

    torch.cuda.set_device(cfg.gpu)
    print('| distributed init (rank {}): {}'.format(
        cfg.rank, cfg.dist_url), flush=True)
    torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                         world_size=cfg.world_size, rank=cfg.rank)
    torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)