# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Adapted from the original SceneTracker transformer; see THIRD_PARTY.md.
"""Only the attention/MLP shapes present in the PCL and DEP checkpoints."""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, count, channels = x.shape
        qkv = self.qkv(x).reshape(batch, count, 3, 8, channels // 8)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.proj(x.transpose(1, 2).reshape(batch, count, channels))


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AttnBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class UpdateFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = nn.Linear(1068, 384)
        self.flow_head = nn.Linear(384, 387)
        self.time_blocks = nn.ModuleList([AttnBlock(384) for _ in range(12)])
        self.space_blocks = nn.ModuleList([AttnBlock(384) for _ in range(12)])

    def forward(self, x):
        x = self.input_transform(x)
        batch, tracks, frames, _ = x.shape
        for time, space in zip(self.time_blocks, self.space_blocks):
            x = time(rearrange(x, 'b n t c -> (b n) t c'))
            x = rearrange(x, '(b n) t c -> b n t c', b=batch, n=tracks)
            x = space(rearrange(x, 'b n t c -> (b t) n c'))
            x = rearrange(x, '(b t) n c -> b n t c', b=batch, t=frames)
        return self.flow_head(x)


class ControlFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = nn.Linear(1068, 256)
        self.output_head = nn.Linear(256, 1)
        self.weight_actvn = nn.LeakyReLU()
        self.time_blocks = nn.ModuleList([AttnBlock(256) for _ in range(6)])
        self.space_blocks = nn.ModuleList([AttnBlock(256) for _ in range(6)])

    def forward(self, x):
        x = self.input_transform(x)
        batch, tracks, frames, _ = x.shape
        for time, space in zip(self.time_blocks, self.space_blocks):
            x = time(rearrange(x, 'b n t c -> (b n) t c'))
            x = space(rearrange(x, '(b n) t c -> (b t) n c', b=batch, n=tracks))
            x = rearrange(x, '(b t) n c -> b n t c', b=batch, t=frames)
        return self.weight_actvn(self.output_head(x))
