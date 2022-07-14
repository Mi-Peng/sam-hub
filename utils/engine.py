import os
import time
from typing import Iterable


import torch
import torch.distributed as dist

from hydra.core.hydra_config import HydraConfig

from solver.utils import Freeze_BN_Running_Stats
from solver.utils import Sync_Perturbation
from utils.configurable import configurable
from utils.dist import is_dist_avail_and_initialized, is_main_process


def _cfg_to_train_one_epoch(cfg):
    return {
        'log_freq': cfg.log_freq,
        'use_closure': cfg.optimizer.get("use_closure", False),
        'single_bn_stats': cfg.optimizer.get("use_closure", False) and cfg.optimizer.get("single_bn_stats", False),
        'sync_perturbation': cfg.optimizer.get("use_closure", False) and cfg.optimizer.get("sync_perturbaton", False),
    }

@configurable(from_config=_cfg_to_train_one_epoch)
def train_one_epoch(
    model: torch.nn.Module,
    train_loader: Iterable,
    criterion, optimizer: torch.optim.Optimizer, lr_scheduler,
    recorder, epoch: int, 
    
    log_freq: int,
    use_closure: bool, single_bn_stats: bool, sync_perturbation: bool,

):
    model.train()

    train_loss_metric = Metric("train_loss")
    train_acc1_metric = Metric("train_acc1")
    train_acc5_metric = Metric("train_acc5")
    metrics = [train_loss_metric, train_acc1_metric, train_acc5_metric]
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = time.time()
        
        lr_scheduler.set_loader_length(len(train_loader))
        lr_scheduler.step(epoch, batch_idx)

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        def closure_first(per_data=False):
            optimizer.zero_grad()
            output = model(images)
            loss_per_data = criterion(output, targets)
            loss = loss_per_data.mean()
            loss.backward()
            return_loss = loss_per_data if per_data else loss
            return return_loss, output
        
        def closure_second(indices=None):
            if indices is None: indices = range(len(targets))
            optimizer.zero_grad()
            with Freeze_BN_Running_Stats(model, single_bn_stats):
                output = model(images[indices])
                loss_per_data = criterion(output, targets[indices])
                loss = loss_per_data.mean()
                loss.backward()

        if use_closure:
            loss, output = optimizer.step(
                closure_first,
                closure_second, 
                sync_perturbation=sync_perturbation,
                epoch=epoch,
                batch_idx=batch_idx,
                global_step=epoch*len(train_loader)+batch_idx,
                model=model, 
                images=images,
                targets=targets,
                criterion=criterion,
                train_data=train_loader.dataset, 
            ) 
        else: 
            loss, output = closure_first()
            optimizer.step()
        
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_num = images.shape[0]
        train_loss_metric.update(loss.item(), n=batch_num)
        train_acc1_metric.update(acc1.item(), n=batch_num)
        train_acc5_metric.update(acc5.item(), n=batch_num)

        msg = ' '.join([
            'Epoch: {epoch}',
            '[{batch_id}/{batch_len}]',
            'lr:{lr:.6f}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Time:{batch_time:.3f}s'
        ])
        if batch_idx % log_freq == 0:
            recorder.info(
                msg.format(
                    epoch=epoch, 
                    batch_id=batch_idx, batch_len = len(train_loader),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=train_loss_metric.global_avg,
                    train_acc1=train_acc1_metric.global_avg,
                    train_acc5=train_acc5_metric.global_avg,
                    batch_time=time.time() - batch_start,
                )
            )
        

    for metric in metrics:
        metric.synchronize_between_processes()
    
    return {
        metric.name: metric.global_avg for metric in metrics
    }




@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: Iterable,
):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    test_loss_metric = Metric("test_loss")
    test_acc1_metric = Metric("test_acc1")
    test_acc5_metric = Metric("test_acc5")
    metrics = [test_loss_metric, test_acc1_metric, test_acc5_metric]

    for images, targets in val_loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        batch_num = images.shape[0]
        test_loss_metric.update(loss.item(), n=batch_num)
        test_acc1_metric.update(acc1.item(), n=batch_num)
        test_acc5_metric.update(acc5.item(), n=batch_num)
    for metric in metrics:
        metric.synchronize_between_processes()

    return {
        metric.name: metric.global_avg for metric in metrics
    }

def accuracy(output, targets, topk=(1,)):
    # output: [b, n]
    # targets: [b]
    batch_size, n_classes = output.size()
    maxk = min(max(topk), n_classes)
    _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t() # pred: [b, maxk] -> [maxk, b]
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred)) # targets: [b] -> [1, b] -> [maxk, b]; correct(bool): [maxk, b]
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class Metric:
    def __init__(self, name):
        self.name = name
        self.value = 0
        self.num = 0
    
    def update(self, value, n=1):
        self.num += n
        self.value += value * n
    
    @property
    def global_avg(self):
        return self.value / self.num

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized(): return 
        t = torch.tensor([self.num, self.value], device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        self.num = int(t[0])
        self.value = t[1]