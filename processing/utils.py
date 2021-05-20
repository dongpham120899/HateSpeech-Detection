import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.)
    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, target, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)


def binary_cross_entropy_with_logits(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=True):
    return binary_cross_entropy(inputs, target, weight, reduction, smooth_eps, from_logits)


class BCELoss(nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target,
                                    weight=self.weight, reduction=self.reduction,
                                    smooth_eps=self.smooth_eps, from_logits=self.from_logits)


class BCEWithLogitsLoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=True):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average,
                                                reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits)
        
class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

def read_csv(path):
    train = pd.read_csv(path + 'train_v7.csv')
    test = pd.read_csv(path + 'test_v7.csv')
    return train, test

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False