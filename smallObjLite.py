from email import policy
from lib2to3.pgen2 import token
from nis import match
from tkinter import W
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, PatchEmbed, Block

from agent import Network
from utils.utils import *
import math

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

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.linear = linear
#         self.sr_ratio = sr_ratio
#         if not linear:
#             if sr_ratio > 1:
#                 self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#                 self.norm = nn.LayerNorm(dim)
#         else:
#             self.pool = nn.AdaptiveAvgPool2d(7) #P = 7; pool(x) -> (B, N, 7, 7)
#             self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
#             self.norm = nn.LayerNorm(dim)
#             self.act = nn.GELU()
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         if not self.linear: #ver1 SRA with Conv
#             if self.sr_ratio > 1:
#                 x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#                 x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#                 x_ = self.norm(x_)
#                 kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             else:
#                 kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else: #linear SRA with pooling
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
#             x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1) #B, P*P, C=dim
#             x_ = self.norm(x_)
#             x_ = self.act(x_)#activation
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             # after kv -> B, P**2, 2*dim; after reshape -> B, P**2*num_of_heads , 2(k&v), head_dim = C//num_of_heads
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # q, k, v shape: B, num_heads, N(num_patches), embed_dim/num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale #B, num_heads, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # attn @ v -> B, num_heads, N, V -> transpose(1,2) -> B, N, num_heads, V -> B, N, num_heads*V=embed_dim
        x = self.proj(x) # B, N, dim
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #x = x + self.drop_path(self.attn(self.norm1(x)))
        tokens = self.attn(self.norm1(x))
        x = x + self.drop_path(tokens)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, tokens

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
        patch_size: kernel size of overlapping embedding. equals to 2*stride-1, must be
        larger than stride size so it could have some overlap parts
    """

    def __init__(self, img_size=512, patch_size=63, stride=32, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        print('***'*10, img_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.stride = stride
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape #H, W = img_size/stride 
        x = x.flatten(2).transpose(1, 2) # x: B,E,H,W -> x.fallten(2): B, E, H*W
        x = self.norm(x)

        return x, H, W


class Smallobj(nn.Module):
    def __init__(self, img_size=512, patch_size=63, embed_dim=512, stride=32, num_stages=3,
                in_chans=3, num_heads=8, mlp_ratio=4, depth=1, num_classes=3, keep_ratio=1/4):
        super().__init__()
        #self.num_classes = num_classes
        #self.depths = depths
        self.keep_ratio = keep_ratio
        self.num_stages = num_stages
        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
        #     for i in range(depth)
        # ])
        self.block = Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
        
        #self.cnn_divider = nn.Conv2d(embed_dim, embed_dim*4, kernel_size=16, stride=16)
        #self.cnn_divider2 = nn.Conv2d(embed_dim*4, embed_dim*16, kernel_size=)
        # patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
        #                         patch_size=7 if i == 0 else 3,
        #                         stride=4 if i == 0 else 2,
        #                         in_chans=in_chans if i == 0 else embed_dims[i - 1],
        #                         embed_dim=embed_dims[i]) #

        #for i in range(num_stages):
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=63, stride=32, in_chans=3, embed_dim=embed_dim)
        # Overlap embedding make the real patch_size is about the same as stride

        # B, C=embed_dim//16, W=input_size/stride//2, H=input_size/stride//2
        len_keep = int((img_size//stride)**2*keep_ratio)
        for i in range(int(len_keep)):
            #smallConv = nn.Conv2d(embed_dim, embed_dim//16, patch_size=stride-1, stride=stride//2,
            #                padding=( (stride-1)//2, (stride-1)//2))
            smallConv = nn.Conv2d(3, embed_dim//16, kernel_size=stride*2, stride=stride*2)
            #B, C=embed_dim//16, 1, 1
            #smallConv = nn.Conv2d(embed)
            setattr(self, f"smallConv{i+1}", smallConv) #B, C, W, H
        self.small_embed_size = embed_dim//16
        self.linearOut = nn.Linear(embed_dim//16*len_keep, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        #under featuremap mode, rewards is sparse
        self.conv2d = nn.Conv2d(embed_dim*16, embed_dim, 1)# -> B, embed_dim, 4, 4
        self.weight1, self.weight2, self.weight3 = torch.Tensor([0.33]).to('cuda'), torch.Tensor([0.33]).to('cuda'), torch.Tensor([0.33]).to('cuda')
        self.rewards = []

        self.loss_func = nn.CrossEntropyLoss()

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def agent_forward(self, tokens, num_tokens, ALPHA=0.8):
        B, N, C = tokens.shape
        self.agent = Network(N, C).to(tokens.device)
        out = self.agent(tokens)
        out = F.sigmoid(out)
        out = out*ALPHA + (1-ALPHA) * (1-ALPHA)

        distr = Bernoulli(out)
        policy_sample = distr.sample()

        return policy_sample, distr
    
    def select_patch(self, x, policy_sample, imgs):
        '''
        x: the input tokens with full length
        policy_sample: the probs of keeping the patch
        '''
        keep_ratio = self.keep_ratio
        B1, N1, C1 = x.shape
        B2, N2, C2 = imgs.shape
        len_keep = int(N1 * (keep_ratio))
        ids_sorted = torch.argsort(policy_sample, dim=1, descending=True)

        #keep the first 1/4, Halve the H, W again
        ids_keep = ids_sorted[:, :len_keep] #
        #x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,C)) # B, len_keep=H*W/4, C
        #self.rewards
        #trnasform x into featureMaps
        # sqrt(N) = H/patch_size, W/patch_size
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C1))
        imgs = torch.gather(imgs, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C2))
        #x = x.reshape(B, C, int((N**0.5)/2), int((N**0.5)/2))# B, C, H/2, W/2
        imgs = imgs.reshape(B2, len_keep, 3, int((C2//3)**0.5), int((C2//3)**0.5))
        imgs = F.interpolate(imgs, scale_factor=(1,2,2)) #B, N, C, H, W
        #x = torch.Conv2d

        return x, imgs, ids_keep

    def agent_loss(self, rewards, distr, policy, gamma=0.99, eps=0.001):
        #returns = torch.Tensor([gamma**2*rewards, gamma*rewards, rewards])
        
        #returns = (returns - returns.mean())/(returns.std() + eps)
        #log_probs = [-distr.log_prob(policies)]
        #log_probs = [-distr.log_prob(p) for p in policies]
        log_probs = -distr.log_prob(policy)
        policy_loss = log_probs * rewards.unsqueeze(1).expand_as(log_probs)
        # for log_prob, R in zip(log_probs, returns):
        #     policy_loss.append(log_prob * R.unsqueeze(1).expand_as(log_prob))
        
        #policy_loss = torch.cat(policy_loss).sum()
        return policy_loss
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.stride #size of one patch (not number of !!)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def forward(self, x, y):
        #Stage 1
        imgs = self.patchify(x)
        y = y.to(x.device)
        #policies = []
        x, H, W = self.patch_embed(x) #H, W = 512/stride = 512/32 = 16
        num_tokens = H*W # 16**2=256
        #assert num_tokens==x.shape[-1]*x.shape[-2], "check the H*W of inputs Xs"
        x, _ = self.block(x)# B, C, 256
        policy_sample, distr = self.agent_forward(x, num_tokens)
        #policies.append(-distr.log_prob(policy_sample))
        x, imgs, ids_keep = self.select_patch(x,policy_sample=policy_sample, imgs=imgs) #B, C, 32, 32
        #imgs = imgs[:, :, ids_keep]
        #Stage 2
        outputs = []
        # for i, idx in enumerate(ids_keep):
        #     smallConv = getattr(self, f"smallConv{i+1}")
        #     res = smallConv(imgs[:,:,idx]) #
        #     outputs.append(res.reshape(imgs.shape[0], self.small_embed_size))

        for i in range(64):
            smallConv = getattr(self, f'smallConv{i+1}')
            res = smallConv(imgs[:,i,:,:,:].squeeze(1))
            outputs.append(res.reshape(imgs.shape[0], self.small_embed_size))


        x = torch.cat(outputs, 1)
        
        #x = self.conv2d(x).reshape(x.shape[0], -1)#B, H_3*W_3*embed_size 
        x = self.linearOut(x)
        preds = self.classifier(x)
        loss = self.loss_func(preds, y)
        #self.rewards.append(compute_rewards(x))
        #in token model, only one reward was received alast
        self.rewards, match = compute_rewards(preds, y)
        policy_loss= self.agent_loss(self.rewards, distr, policy_sample)

        return loss, policy_loss.mean(), match.float(), self.rewards

        


        

        



        
