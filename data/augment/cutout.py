'''
Cutout:
    A data augmentation: Randomly mask out a patch from the image.
paper: https://arxiv.org/abs/1708.04552, arXiv2017
github: https://github.com/uoguelph-mlrg/Cutout
'''
import numpy as np
import torch


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, image):
        """
        Args:
            image(Tensor): (C, H, W)
        """
        if np.random.rand(1) > self.p: return image

        h, w = image.size(1), image.size(2)
        y, x = np.random.randint(h), np.random.randint(w)

        left = np.clip(x - self.size // 2, 0, w)
        right = np.clip(x + self.size // 2, 0, w)
        top = np.clip(y - self.size // 2, 0, h)
        bottom = np.clip(y + self.size // 2, 0, h)

        image[:, top: bottom, left: right] = 0.
        return image