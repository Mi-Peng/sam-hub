
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.configurable import configurable

class LabelSmoothCrossEntropy(nn.Module):
    @configurable
    def __init__(self, smoothing=0., num_classes=None, reduction="mean"):
        super(LabelSmoothCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.num_classes = num_classes

        assert reduction in ["mean", "sum", "none"]
        self.reduction=reduction

    @classmethod
    def from_config(cls, cfg):
        return {
            "smoothing": cfg.smoothing,
            "num_classes": cfg.data.dataset.n_classes,
            "reduction": "none",
        }

    def forward(self, x, labels):
        '''
        Args:
            x: 
                output of model, size: [B, N]
            labels:
                ground truth of x, size: [B] or [B, N]
        '''
        if labels.ndim == 1: 
            # labels: [B]
            labels = labels.long().view(-1, 1) # labels [B, 1]
            labels = torch.full((labels.size(0), self.num_classes), 0., device=labels.device).scatter_(1, labels, 1.0)
        elif labels.ndim == 2:
            # labels: [B, N]
            assert labels.size(1) == self.num_classes
        else:
            raise ValueError
        labels = labels * (1 - self.smoothing) + self.smoothing / self.num_classes # labels: [B, N]
        log_probs = F.log_softmax(x, dim=-1)
        loss = torch.sum(- labels * log_probs, dim=-1)

        if self.reduction == "sum": 
            loss = loss.sum()
        elif self.reduction == "mean": 
            loss = loss.mean()
        return loss

# B = 4,
# N = 10
# loss_fn = LabelSmoothCrossEntropy(smoothing=0.1, num_classes=N, reduction="none")
# output = torch.rand(B, N)
# labels = torch.randint(0, N, size=(B,))
# loss = loss_fn(output, labels)
# print(loss)