"""ConvONet triplane encoder; original attribution is in THIRD_PARTY.md."""
import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_max
from .unet import UNet


class ResnetBlockFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_0 = nn.Linear(256, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.actvn = nn.ReLU()
        self.shortcut = nn.Linear(256, 128, bias=False)

    def forward(self, x):
        residual = self.fc_1(self.actvn(self.fc_0(self.actvn(x))))
        return self.shortcut(x) + residual


def plane_index(points, axes):
    xy = torch.clamp(points[:, :, axes], min=1e-8, max=1. - 1e-8)
    xy = (xy * 256).long()
    return (xy[:, :, 0] + 256 * xy[:, :, 1])[:, None, :]


class LocalPoolPointnet(nn.Module):
    def __init__(self, checkpoint_global_head=False):
        super().__init__()
        self.fc_pos = nn.Linear(3, 256)
        self.blocks = nn.ModuleList([ResnetBlockFC() for _ in range(5)])
        self.fc_c = nn.Linear(128, 128)
        self.unet = UNet()
        if checkpoint_global_head:
            # Stored by the original tracker, though its forward never uses it.
            # Retain these two tensors so the entire checkpoint loads strictly.
            self.global_feat_head = nn.Linear(128, 128)

    def forward(self, points):
        points = (points + 1.) / 2.
        # Preserve the original pooling addition order: xz, xy, yz.
        indices = {plane: plane_index(points, axes) for plane, axes in
                   [('xz', [0, 2]), ('xy', [0, 1]), ('yz', [1, 2])]}
        features = self.blocks[0](self.fc_pos(points))
        for block in self.blocks[1:]:
            pooled = 0
            for index in indices.values():
                plane = scatter_max(features.permute(0, 2, 1), index, dim_size=256 ** 2)[0]
                pooled = pooled + plane.gather(2, index.expand(-1, 128, -1))
            features = block(torch.cat([features, pooled.permute(0, 2, 1)], dim=2))
        features = self.fc_c(features).permute(0, 2, 1)
        planes = {}
        for key, index in indices.items():
            plane = features.new_zeros(points.shape[0], 128, 256 ** 2)
            plane = scatter_mean(features, index, out=plane).reshape(-1, 128, 256, 256)
            planes[key] = self.unet(plane)
        return torch.stack([planes['xy'], planes['xz'], planes['yz']]).permute(1, 0, 2, 3, 4)
