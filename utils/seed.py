import torch
import numpy as np
import random

from utils.dist import get_rank

def setup_seed(cfg):
    if cfg.seed is None: return
    seed = cfg.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)