import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def target_format(target, loss_mode, device):
    if loss_mode == 'mse':
        return target.to(device, dtype=torch.float)
    elif loss_mode == 'ce':
        return target.to(device, dtype=torch.long)

def get_acc(x, target, loss_mode):
    if loss_mode == 'mse':
        pred = x.data.max(1, keepdim=True)[1]
        tar = target.data.max(1, keepdim=True)[1]
        return pred.eq(tar.data).sum().item()
    if loss_mode == 'ce':
        pred = x.data.max(1, keepdim=True)[1]
        return pred.eq(target.data.view_as(pred)).sum().item()
