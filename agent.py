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

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, patch_size, in_chans, embed_dim, stride=1): #in_planes=64, planes = 64, 128, 256, 512...
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential( #use 1*1 kernel to upsize the input 
#                                           #so it can be added with output with self.expansion*planes size
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Agent(nn.Module):
    def __init__(self, N, C):
        super().__init__()
        self.mlp = Mlp(N*C, 2*N*C, N)
        # self.linearProj = nn.Linear(token_in_dim, token_out_dim)
        # self.linearInter = nn.Linear(out_size+token_out_dim*token_num, inter_dim)
        # self.linearOut = nn.Linear(inter_dim, num_classes)
        # self.embedding = nn.Linear()
        
    def forward(self, tokens, H, W):
        B, N, C = tokens.shape #C = embed_dim
        out = self.mlp(tokens.reshape(B, N*C)) #B, N
        # imgs = self.resnet(imgs) # [B, 512]
        # tokens = self.linearProj(tokens) #[B, L, 512]
        # tokens =  torch.reshape(tokens, (tokens.shape[0], -1)) #[b, L*64]
        # #print(f'tokens: {tokens.shape}')
        # flat_input = torch.cat((imgs, tokens), dim=1) 
        # #print('flat:',flat_input.shape)
        # flat_input = self.linearInter(flat_input)
        #out = self.linearOut(flat_input) #[B, 196]

        return out


        


    