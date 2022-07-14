'''
Mixup:
    A data augmentation: Each pixel of the input image is fused proportionally, and the output result is fused proportionally.
paper: https://arxiv.org/abs/1710.09412, accepted by ICLR2018 
github: https://github.com/facebookresearch/mixup-cifar10
'''
import numpy as np
import torch

class Mixup:
    def __init__(self, alpha=1.0, p=0.5, num_classes=None):
        self.alpha = alpha
        self.p = p
        self.num_classes = num_classes
    
    def __call__(self, image, target):
        """
        Args:
            image(Tensor): (B, C, H, W)
        """
        if np.random.rand(1) > self.p: return image, 1.0
        lambda_mixup = np.random.beta(self.alpha, self.alpha)
        
        image_flipped = image.flip(0).mul_(1. - lambda_mixup)
        image.mul_(lambda_mixup).add_(image_flipped)

        target = mixup_target(target, self.num_classes, lambda_mixup)
        return image, target, lambda_mixup


def mixup_target(target, num_classes, lambda_mixup):
    '''
    Args:
        target: [B]
    Return:
        lambda * target1 + (1 - lambda) * target2 : [B, N]
    '''
    target1 = target.long().view(-1, 1) # target1: [B, 1]
    target1 = torch.full((target1.size(0), num_classes), 0.0, device=target1.device).scatter_(1, target1, 1.0) # target1: [B, N]
    
    target2 = target.flip(0).long().view(-1, 1) # target2: [B, 1]
    target2 = torch.full((target2.size(0), num_classes), 0.0, device=target2.device).scatter_(1, target2, 1.0) # target2: [B, N]
    return lambda_mixup * target1 + (1 - lambda_mixup) * target2