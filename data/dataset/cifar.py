import numpy as np
import torchvision.datasets

from utils.configurable import configurable

from data.build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CIFAR10:
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
        train_data = torchvision.datasets.CIFAR10(root=self.datapath, train=True, transform=self.train_transform, download=True)
        val_data = torchvision.datasets.CIFAR10(root=self.datapath, train=False, transform=self.test_transform, download=True)
        return train_data, val_data


@DATASET_REGISTRY.register()
class CIFAR100:
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
        train_data = torchvision.datasets.CIFAR10(root=self.datapath, train=True, transform=self.train_transform, download=True)
        val_data = torchvision.datasets.CIFAR10(root=self.datapath, train=False, transform=self.test_transform, download=True)
        return train_data, val_data
