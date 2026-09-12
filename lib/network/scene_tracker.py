"""Inference-only architecture shared by the PCL and DEP checkpoints."""
from pathlib import Path
import torch
from torch import nn
from .feature_net.feature_net import LocalPoolPointnet
from .deformer import Deformer
from .corr_block import CorrBlock
from .update_former import UpdateFormer, ControlFormer
from .refinement import refine
from .util import sample1_bilinear


class SceneTracker(nn.Module):
    window = 8

    def __init__(self):
        super().__init__()
        self.deformer = Deformer()
        self.feature_net = LocalPoolPointnet(checkpoint_global_head=True)
        self.update_former = UpdateFormer()
        # The trailing underscore is part of the original checkpoint keys.
        self.control_former_ = ControlFormer()
        self.corr_block = CorrBlock()
        self.track_feat_updater = nn.Sequential(nn.GroupNorm(1, 128), nn.Linear(128, 128), nn.GELU())

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cuda'):
        """Load all 652 entries, including the embedded deformer, strictly."""
        model = cls()
        # These trusted project checkpoints also serialize optimizer/scheduler objects.
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=False, mmap=True)
        if state.get('dict_version') != '1.0':
            raise ValueError('Expected a SceneTracker v1.0 checkpoint')
        model.load_state_dict(state['model_state_dict'], strict=True)
        return model.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, points, reference, controls, progress=None):
        """Normalized inputs: [1,F,P,3], [1,N,3], [1,T,3]; returns [1,F,T,4].

        Every control is queried in the reference frame. Output channels are
        xyz followed by the learned radius used for mesh blending.
        """
        if points.ndim != 4 or points.shape[0] != 1 or points.shape[-1] != 3:
            raise ValueError('points must have shape [1, frames, points, 3]')
        if points.shape[1] < 1 or reference.ndim != 3 or controls.ndim != 3:
            raise ValueError('A nonempty sequence, reference points and controls are required')
        frames, count = points.shape[1], controls.shape[1]
        init = []
        for frame in range(frames):
            init.append(self.deformer(points[:, frame], reference, controls)[:, None])
            if progress and ((frame + 1) % 20 == 0 or frame == frames - 1):
                progress(f'initialized {frame + 1}/{frames} frames')
        init = torch.cat(init, dim=1)
        # Preserve the skin branch's equal-first-frame sort and feature ordering.
        order = torch.sort(torch.zeros(count, device=points.device, dtype=torch.long))[1]
        inverse_order = torch.argsort(order)
        coords = self._pad(init[:, :self.window], self.window)[:, :, order].clone()
        initial_features = sample1_bilinear(self.feature_net(reference), controls)
        initial_features = initial_features.unsqueeze(1).repeat(1, self.window, 1, 1, 1)
        result = points.new_zeros((1, frames, count, 4))
        planes = None
        previous = None
        for start in range(0, max(frames - self.window // 2, 1), self.window // 2):
            window_points = points[:, start:start + self.window]
            valid = window_points.shape[1]
            window_points = self._pad(window_points, self.window)[0]
            if planes is None:
                planes = self.feature_net(window_points)
            else:
                planes = torch.cat([planes[4:], self.feature_net(window_points[4:])], dim=0)
                coords[:, :4] = previous[:, 4:]
                # Keep the original branch's deformer order for newly entering frames.
                coords[:, 4:] = self._pad(init[:, start + 4:start + 8], 4)
            previous, radii = refine(self, planes[None], coords, initial_features, predict_radii=True)
            result[:, start:start + valid, :, :3] = previous[:, :valid]
            result[:, start:start + valid, :, 3:] = radii[:, :, :valid].permute(0, 2, 1, 3)
            if progress and (start // 4 + 1) % 5 == 0:
                progress(f'tracking window {start // 4 + 1}')
        self.corr_block.flush()
        return result[:, :, inverse_order]

    @staticmethod
    def _pad(sequence, length):
        if sequence.shape[1] < length:
            sequence = torch.cat([sequence, sequence[:, -1:].repeat(1, length - sequence.shape[1], 1, 1)], dim=1)
        return sequence
