
import os
import time
import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf

import torch

from data.build import build_dataset, build_train_dataloader, build_val_dataloader
from models.build import build_model
from solver.build import build_optimizer, build_lr_scheduler

from utils.labelsmoothloss import LabelSmoothCrossEntropy
from utils.engine import train_one_epoch, evaluate
from utils.dist import init_distributed_model, is_main_process
from utils.seed import setup_seed
from utils.recorder import Recorder, save_checkpoint, save_model

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: OmegaConf):
    # init dist
    init_distributed_model(cfg)

    # fix seed
    setup_seed(cfg)

    # init recorder
    recorder = Recorder(logger=logger, cfg=cfg)
    cfg_show = OmegaConf.to_yaml(cfg)
    recorder.info('\n' + cfg_show)

    # build dataset
    train_data, val_data = build_dataset(cfg)
    train_loader = build_train_dataloader(train_dataset=train_data, cfg=cfg)
    val_loader = build_val_dataloader(val_dataset=val_data, cfg=cfg)
    recorder.info(
        f"DataSet:{cfg.data.dataset.name} Train Data:{len(train_data)} Test Data: {len(val_data)}."
    )
    
    # build models
    model = build_model(cfg)
    model_without_ddp = model
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        if cfg.syncbn: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(cfg.gpu)
        model_without_ddp = model.module
    recorder.info(
        f'Model: {cfg.model.name}'
    )

    # build solver
    optimizer, base_optimizer = build_optimizer(cfg, model=model_without_ddp)
    lr_scheduler = build_lr_scheduler(cfg, optimizer=base_optimizer)
    recorder.info(
        f'Optimizer: {type(optimizer)}.'
    )
    recorder.info(
        f"LR Scxheduler: {type(lr_scheduler)}."
    )

    # build loss
    criterion = LabelSmoothCrossEntropy(cfg)

    # resume
    start_epoch = 0
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        lr_scheduler.step(start_epoch)
        recorder.info(
            f'Resume training from {cfg.resmue}.'
        )

    # train script
    recorder.info(
        f"Start training for {cfg.epochs} Epochs."
    )
    start_training = time.time()
    max_acc = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            recorder=recorder,
            epoch=epoch,
            cfg=cfg, 
        )
        val_stats = evaluate(
            model=model,
            val_loader=val_loader,
        )
        
        if max_acc < val_stats["test_acc1"]:
            max_acc = val_stats["test_acc1"]
            recorder.wandb_summary(key="best_test_acc1", value=max_acc)
            save_model(model_without_ddp, 'checkpoint.pth')
        
        if (cfg.checkpoint_freq is not None) and (epoch % cfg.checkpoint_freq == 0):
            save_checkpoint({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': cfg,
                }, os.path.join(HydraConfig.get().run.dir, 'checkpoint_{}.pth'.format(epoch)))

        msg = ' '.join([
            'Epoch:{epoch}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Test Loss:{test_loss:.4f}',
            'Test Acc1:{test_acc1:.4f}(Max:{max_acc:.4f})',
            'Test Acc5:{test_acc5:.4f}',
            'Time:{epoch_time:.3f}s'])
        recorder.info(msg.format(epoch=epoch, **train_stats, **val_stats, max_acc=max_acc, epoch_time=time.time()-epoch_start))
        recorder.wandb_log(epoch=epoch, **train_stats)
        recorder.wandb_log(epoch=epoch, **val_stats)
        
    end_training = time.time()
    used_training = str(datetime.timedelta(seconds=end_training-start_training))
    recorder.info('Training Time:{}'.format(used_training))
    recorder.complete_file_w_acc(max_acc)


if __name__ == '__main__':
    main()