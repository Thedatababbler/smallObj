import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T



def compute_rewards(preds, targets, policy, penalty=0, distr=None, id_keeps=None):
    
    patch_use = policy.sum(1).float() / policy.size(1)
    sparse_reward = 1.0 - patch_use**2
    _, pred_idx = preds.max(1)
    match = (pred_idx==targets).data
    reward = sparse_reward#torch.ones(targets.shape)
    reward[~match] = penalty
    reward = reward.to(preds.device)
    return reward, match

def get_transform(name='plane'):
    if name == 'plane':
        normalize = T.Normalize(mean=[0.4205, 0.4205, 0.4205], std=[0.2131, 0.2131, 0.2131])
        transform = T.Compose([
            T.RandomResizedCrop(512),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    if name == 'bird':
        pass

    return transform


def get_dataset(name, root='/ziyuanqin/projects/smallobj/train_data'):
    transform = get_transform()
    if name == 'plane':
        datasets = ImageFolder(root, transform) 
    if name == 'bird':
        root = 'images_bird'
        datasets = ImageFolder(root, transform)
    return datasets

def get_model(model):
    import resnet_cifar as resnet
           
    rnet_hr = resnet.ResNet(resnet.BasicBlock, [3,4,6,3], 3, 10)
    rnet_lr = resnet.ResNet(resnet.BasicBlock, [3,4,6,3], 3, 10)
    agent = resnet.ResNet(resnet.BasicBlock, [1,1,1,1], 3, 256)
