import numpy as np
import torch#.tensor as tensor
import torch.utils.data as torchdata
from torchvision import transforms as T
import utils


# data_loader_train = torch.utils.data.DataLoader(
#     trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
trainset = utils.get_dataset('bird')
loader = torchdata.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)
data = next(iter(loader))
mean, std = data[0].mean(), data[0].std()

print(f'mean={mean}, std={std}')
torchdata.DataLoader