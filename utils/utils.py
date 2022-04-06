import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

def compute_rewards(preds, targets, penalty=-1, distr=None, id_keeps=None):
    
    _, pred_idx = preds.max(1)
    match = (pred_idx==targets).data
    reward = torch.ones(targets.shape)
    reward[~match] = penalty
    reward = reward.to(preds.device)
    return reward, match

def get_transform():
    normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    transform = T.Compose([
        T.RandomResizedCrop(512),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    return transform


def get_dataset(name, root='/ziyuanqin/projects/smallobj/train_data'):
    transform = get_transform()
    if name == 'plane':
        datasets = ImageFolder(root, transform)
    return datasets




