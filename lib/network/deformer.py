"""Pairwise reference-to-frame initialization, loaded from the full checkpoint."""
import torch
from torch import nn
from .feature_net.feature_net import LocalPoolPointnet
from .corr_block import CorrBlock
from .update_former import UpdateFormer
from .refinement import refine
from .util import sample1_bilinear


class Deformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_net = LocalPoolPointnet()
        self.update_former = UpdateFormer()
        self.corr_block = CorrBlock()
        self.track_feat_updater = nn.Sequential(nn.GroupNorm(1, 128), nn.Linear(128, 128), nn.GELU())

    def forward(self, points, reference, controls):
        ref_planes = self.feature_net(reference)
        target_planes = self.feature_net(points)
        features = sample1_bilinear(ref_planes, controls)
        coords, _ = refine(self, torch.stack([ref_planes, target_planes], dim=1),
                           torch.stack([controls, controls.clone()], dim=1),
                           torch.stack([features, features.clone()], dim=1))
        self.corr_block.flush()
        return coords[:, -1]
