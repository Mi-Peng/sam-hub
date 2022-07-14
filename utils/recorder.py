import os
import time
try:
    import wandb
    _import_wandb=True
except:
    _import_wandb=False

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch


from utils.configurable import configurable
from utils.dist import is_dist_avail_and_initialized, is_main_process

class Recorder:
    @configurable
    def __init__(self,
        logger,
        *,
        enable_wandb,
        project,
        name,
        group,
        distributed,
        cfg,
    ) -> None:
        self.logger = logger
        self.enable_wandb = enable_wandb and _import_wandb

        if self.enable_wandb:
            wandb_dict = {
                'project': project,
                'name': name,
                'group': group,
            }
            # if distributed: wandb_dict['group']='DDP'
            if is_main_process():
                self.wandb_run = wandb.init(**wandb_dict, config=OmegaConf.to_container(cfg))

    @classmethod
    def from_config(cls, cfg):
        return {
            "enable_wandb": cfg.wandb.enable_wandb,
            "project": cfg.wandb.project,
            "name": cfg.wandb.name,
            "group": cfg.wandb.group,
            "distributed": cfg.distributed,
            "cfg": cfg,
        }

    @is_main_process
    def info(self, info):
        self.logger.info(info)
    
    @is_main_process
    def wandb_log(self, **adict):
        if self.enable_wandb:
            self.wandb_run.log(adict)
        else:
            return

    @is_main_process
    def wandb_summary(self, key, value):
        if self.enable_wandb:
            wandb.run.summary[key]=value
        else:
            return

    @is_main_process
    def complete_file_w_acc(self, acc):
        os.system('mv {} {}_{:.4f}'.format(HydraConfig.get().run.dir, HydraConfig.get().run.dir, acc))


@is_main_process
def save_checkpoint(state_dict, filename):
    torch.save({
        **state_dict,
    }, os.path.join(HydraConfig.get().run.dir, filename))

@is_main_process
def save_model(model, filename):
    torch.save({
        'model': model.state_dict(),
    }, os.path.join(HydraConfig.get().run.dir, filename))