import numpy as np
import torchvision.datasets 
import torchvision.transforms

from utils.configurable import configurable

from data.build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNet:
    @configurable
    def __init__(self, datapath, mean, std, transform) -> None:
        self.datapath = datapath
        self.mean, self.std = mean, std
        self.train_transform, self.test_transform = transform

    @classmethod
    def from_config(cls, cfg):
        return {
            "datapath": cfg.data.dataset.datapath,
            "mean": cfg.data.dataset.mean,
            "std": cfg.data.dataset.std,
        }
    
    def get_data(self):
        train_dataset = torchvision.datasets.ImageFolder(root=self.datapath + '/train', transform=self.train_transform)
        val_dataset = torchvision.datasets.ImageFolder(root=self.datapath + '/val', transform=self.test_transform)
        return train_dataset, val_dataset
