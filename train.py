from email.policy import default
import datetime
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim.optim_factory as optim_factory
from pathlib import Path
from utils.utils import *
from smallObjLite import Smallobj
from train_one_epoch import train_one_epoch




def get_args_parser():
    parser = argparse.ArgumentParser('smallobj training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--data_path', default='../datasets/tiny_set', type=str,
                        help='dataset root path')
    parser.add_argument('--log_path', default='../logs', type=str)
    parser.add_argument('--output_dir', default='../outputs/trial1', type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--load', default=None, help='checkpoint path')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)

    ## optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    return parser

def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    log_writer = SummaryWriter(log_dir=args.log_path)

    trainset = get_dataset('plane')

    data_loader_train = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    

    if args.load:
        '''
        checkpoint = torch.load(args.load)
        agent.load_state_dict(checkpoint['agent'])
        print('load agent from', args.load)
        '''
        pass

    #model = smallObj_model.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    #model.to(device)
    model = Smallobj().to(args.device)
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    #if args.gpu > 0

    ### Start Training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, data_loader_train, optimizer, args.device, epoch, log_writer, args)
    
        total_time = time.time() - start_time
        total_time_str =str(datetime.timedelta(seconds = int(total_time)))
        print('Training time {}'.format(total_time_str))

        output_dir = Path(args.output_dir)
        epoch_name = str(epoch)
        # checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        # for checkpoint_path in checkpoint_paths:
        #     to_save = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'epoch': epoch,
        #         #'scaler': loss_scaler.state_dict(),
        #         'args': args,
        #     }


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    #if args.output_dir:
    #    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)