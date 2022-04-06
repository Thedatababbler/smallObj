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
    matches, losses, rewards = [], [], []

    #accum_iter = args.accum_iter

    #optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (samples, targets) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
    
        #data_iter_step % accum_iter ==0:
        samples = samples.to(device, non_blocking=True)

        #with torch.cuda.amp.autocast():
        loss1, loss2, loss3, match, reward = model(samples, targets)

        loss = loss1 + loss2 + loss3
        loss_value = loss.item()
        #loss_value2 = loss2.item()
        #loss_value3 = loss3.item()
        optimizer.zero_grad()
        loss.backward()
        #policy_loss.backward()
        optimizer.step()
        matches.append(match.cpu())
        #losses.append(loss.cpu())
        #rewards.append(reward.cpu())
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss, epoch_1000x)
            log_writer.add_scalar('loss1', loss1, epoch_1000x)
            log_writer.add_scalar('loss2', loss2, epoch_1000x)
            log_writer.add_scalar('loss3', loss3, epoch_1000x)
            log_writer.add_scalar('reward', reward.mean(), epoch_1000x)

    acc = torch.cat(matches, 0).mean()
    #loss_reduce = torch.cat(loss, 0).mean()
    #reward_reduce = torch.cat(reward, 0).mean()
    #print(f'Loss: {loss_reduce}, Rewards: {reward_reduce}, Acc: {acc}, Epoch: {epoch}')     
    print(f'Loss: {loss}, Acc: {acc}, Epoch: {epoch}')   
        

        
        

    