import math
from statistics import mode
import sys
from typing import Iterable
from unittest import TestLoader

import torch
import torch.nn.functional as F
#from utils.utils import MetricLogger
import tqdm

def train_one_epoch(model: torch.nn.Module, data_loader:Iterable, optimizer:torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_writer=None, args=None):

    model.train(True)
    
    #metric_logger = MetricLogger(delimiter=" ")
    #metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #header = 'Epoch: [{}}'.format(epoch)
    print_freq = 20
    matches = []
    #accum_iter = args.accum_iter

    #optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (samples, targets) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
    
        #data_iter_step % accum_iter ==0:
        samples = samples.to(device, non_blocking=True)

        #with torch.cuda.amp.autocast():
        loss, match = model(samples, targets)

        loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        #policy_loss.backward()
        optimizer.step()
        matches.append(match.item())

    print(f'Loss: {loss_value}, Acc: { matches.mean()}, Epoch: {epoch}')     
        

        
        

    