"""Four-level triplane correlation with unit feature weights."""
import torch
from torch import nn
from torch.nn import functional as F

def bilinear_sampler(img, coords):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    return img

class CorrBlock(nn.Module):

    def __init__(self, num_levels=4, radius=4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.feat_init = False

    def init(self, feature_planes):
        B, S, _, C, HW = feature_planes.shape[:5]
        self.S = S
        self.feature_planes_pyramid = []
        self.feature_planes_pyramid.append(feature_planes)
        f_planes = feature_planes
        for _ in range(self.num_levels - 1):
            f_planes = f_planes.reshape(B * S * 3, C, HW, HW)
            f_planes = F.avg_pool2d(f_planes, 2, stride=2)
            _, _, HW, _ = f_planes.shape
            self.feature_planes_pyramid.append(f_planes.reshape(B, S, 3, C, HW, HW))
        self.feat_init = True

    def reinit(self, feature_planes: torch.Tensor):
        B, S, _, C, HW = feature_planes.shape[:5]
        assert S == self.S
        feature_planes = feature_planes[:, S // 2:]
        second_half_fpp = []
        second_half_fpp.append(feature_planes)
        f_planes = feature_planes
        for _ in range(self.num_levels - 1):
            f_planes = f_planes.reshape(B * (S // 2) * 3, C, HW, HW)
            f_planes = F.avg_pool2d(f_planes, 2, stride=2)
            _, _, HW, _ = f_planes.shape
            second_half_fpp.append(f_planes.reshape(B, S // 2, 3, C, HW, HW))
        for i in range(self.num_levels):
            f_planes_i = self.feature_planes_pyramid[i]
            self.feature_planes_pyramid[i] = torch.cat([f_planes_i[:, S // 2:], second_half_fpp[i]], dim=1)
            assert self.feature_planes_pyramid[i].shape[1] == self.S

    def flush(self):
        self.feat_init = False
        self.feature_planes_pyramid = []

    def corr(self, track_feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, S, T, _, C = track_feats.shape
        r = self.radius
        track_feats = track_feats.reshape(B * S * T, 3, C)
        corr_xy = []
        corr_xz = []
        corr_yz = []
        for i in range(self.num_levels):
            f_map = self.feature_planes_pyramid[i]
            f_hw = f_map.shape[-1]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords).reshape(-1, 2)
            l1 = lambda x: torch.abs(x[:, 0]) + torch.abs(x[:, 1])
            delta = delta[l1(delta) <= r + 0.001]
            centroid_lvl = (coords.reshape(B * S, T, 3) + 1.0) / 2.0 * f_map.shape[-1]
            centroid_lvl_xy = centroid_lvl[..., [0, 1]]
            centroid_lvl_xz = centroid_lvl[..., [0, 2]]
            centroid_lvl_yz = centroid_lvl[..., [1, 2]]
            delta_lvl = delta.view(1, 1, -1, 2)
            coords_lvl_xy = centroid_lvl_xy[:, :, None, :] + delta_lvl
            coords_lvl_xz = centroid_lvl_xz[:, :, None, :] + delta_lvl
            coords_lvl_yz = centroid_lvl_yz[:, :, None, :] + delta_lvl
            feat_xy = bilinear_sampler(f_map[:, :, 0].reshape(B * S, C, f_hw, f_hw), coords_lvl_xy)
            feat_xz = bilinear_sampler(f_map[:, :, 1].reshape(B * S, C, f_hw, f_hw), coords_lvl_xz)
            feat_yz = bilinear_sampler(f_map[:, :, 2].reshape(B * S, C, f_hw, f_hw), coords_lvl_yz)
            feat_xy = feat_xy.permute((0, 2, 3, 1)).reshape(B * S * T, -1, C)
            feat_xz = feat_xz.permute((0, 2, 3, 1)).reshape(B * S * T, -1, C)
            feat_yz = feat_yz.permute((0, 2, 3, 1)).reshape(B * S * T, -1, C)
            corr_xy.append(torch.matmul(feat_xy, track_feats[:, 0, :, None]).reshape(B, S, T, -1))
            corr_xz.append(torch.matmul(feat_xz, track_feats[:, 1, :, None]).reshape(B, S, T, -1))
            corr_yz.append(torch.matmul(feat_yz, track_feats[:, 2, :, None]).reshape(B, S, T, -1))
        corr_xy = torch.cat(corr_xy, dim=-1)
        corr_xz = torch.cat(corr_xz, dim=-1)
        corr_yz = torch.cat(corr_yz, dim=-1)
        return torch.cat([corr_xy, corr_xz, corr_yz], dim=-1) / C ** 0.5
