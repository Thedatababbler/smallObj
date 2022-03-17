from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from timm.models.vision_transformer import PatchEmbed

#ALPHA = 0.8
EPSILON = 0.9 #greedy policy
GAMMA = 0.9 #discount rate

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, patch_size, in_chans, embed_dim, stride=1): #in_planes=64, planes = 64, 128, 256, 512...
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential( #use 1*1 kernel to upsize the input 
                                          #so it can be added with output with self.expansion*planes size
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_kernel_size, out_size):
        #passed block is a class not an instance
        # num_blocks is a list like following: [3,4,6,3]
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=initial_kernel_size, stride=7, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2) #num_blocks[0] * 64-channel 3*3 kernel conv-layers 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) #num_blocks[1] * 128-channel stride=2 downsampling 3*3 kernel conv-layers 
        #self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = nn.Linear(128*block.expansion, out_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [2]*(num_blocks-1) # [stride, 1, 1, ... num_blocks-1*1]
        #only first step use stride to downsample 
        layers = []
        for stride in strides:
            #print(f'stride: {stride} fro [[M')
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, res='lr'):
        #x = torch.nn.functional.interpolate(x, (256, 256), mode='bicubic')
        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out) 
        out = self.layer1(out)
        #print('block1_out', out.shape)
        out = self.layer2(out)
        #print('block2_out', out.shape)
        #out = self.layer3(out)
        # print('block3_out', out.shape)
        # out = self.layer4(out)
        # print('block4_out', out.shape)
        if res == 'lr':
            #print(out.shape)
            out = F.avg_pool2d(out, 1)
            #print(out.shape)
        else:
           out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(out)
        #print(out.shape)
        return out

class Network(nn.Module):
    def __init__(self, token_in_dim, token_num, out_size=256, token_out_dim=16, 
                inter_dim = 1024, num_classes = 196):
        super().__init__()

        self.resnet = ResNet(BasicBlock, [2, 3, 1, 1], 7, out_size)
        self.linearProj = nn.Linear(token_in_dim, token_out_dim)
        self.linearInter = nn.Linear(out_size+token_out_dim*token_num, inter_dim)
        self.linearOut = nn.Linear(inter_dim, num_classes)
        self.embedding = nn.Linear()
        
    def forward(self, imgs):
        tokens = self.patch_embed(imgs)
        imgs = self.resnet(imgs) # [B, 512]
        tokens = self.linearProj(tokens) #[B, L, 512]
        tokens =  torch.reshape(tokens, (tokens.shape[0], -1)) #[b, L*64]
        #print(f'tokens: {tokens.shape}')
        flat_input = torch.cat((imgs, tokens), dim=1) 
        #print('flat:',flat_input.shape)
        flat_input = self.linearInter(flat_input)
        out = self.linearOut(flat_input) #[B, 196]

        return out


        


    