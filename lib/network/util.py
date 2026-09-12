# Bilinear sampling adapted from Meta CoTracker; see THIRD_PARTY.md.
import numpy as np
import torch

def get_3d_embedding(xyz, C):
    B, N, D = xyz.shape
    assert D == 3
    x = xyz[:, :, 0:1]
    y = xyz[:, :, 1:2]
    z = xyz[:, :, 2:3]
    div_term = (torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))
    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2)
    return pe.to(xyz)

def positional_encoding(x: torch.Tensor, embed_dim) -> torch.Tensor:
    assert embed_dim % 6 == 0
    B, S, T = x.shape[:3]
    x = x.reshape(-1)
    embed_dim //= 3
    omega = torch.arange(embed_dim // 2, dtype=torch.float).to(x)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = torch.einsum('m,d->md', x, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.reshape(B, S, T, -1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def bilinear_sample2d(im, x, y):
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]
    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H
    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])
    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2
    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip
    im_flat = im.permute(0, 2, 3, 1).reshape(B * H * W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)
    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    return output

def sample1_bilinear(feature_planes: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    hw_feat = feature_planes.shape[-1]
    coords0 = (coords + 1.0) / 2.0 * hw_feat
    x_coord = coords0[..., 0]
    y_coord = coords0[..., 1]
    z_coord = coords0[..., 2]
    xy_output = bilinear_sample2d(feature_planes[:, 0], x_coord, y_coord).permute((0, 2, 1))
    xz_output = bilinear_sample2d(feature_planes[:, 1], x_coord, z_coord).permute((0, 2, 1))
    yz_output = bilinear_sample2d(feature_planes[:, 2], y_coord, z_coord).permute((0, 2, 1))
    output = torch.stack([xy_output, xz_output, yz_output]).permute((1, 2, 0, 3))
    return output
