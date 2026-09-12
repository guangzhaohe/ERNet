"""Shared six-step coordinate refinement for the deformer and tracker."""
import torch
from .util import get_3d_embedding, positional_encoding, get_1d_sincos_pos_embed_from_grid


def refine(module, planes, initial_coords, initial_features, predict_radii=False):
    batch, frames, tracks = initial_coords.shape[:3]
    coords = initial_coords.clone()
    features = initial_features.clone()
    if module.corr_block.feat_init:
        module.corr_block.reinit(planes)
    else:
        module.corr_block.init(planes)
    position = positional_encoding(coords, 1068).permute(0, 2, 1, 3).reshape(batch * tracks, frames, -1)
    times = torch.linspace(0, frames - 1, frames).reshape(1, frames, 1)
    temporal = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(1068, times[0]))
    temporal = temporal[None].repeat(batch, 1, 1).float().to(coords)
    radii = None
    for _ in range(6):
        corr = module.corr_block.corr(features, coords)
        corr = corr.permute(0, 2, 1, 3).reshape(batch * tracks, frames, -1)
        flow = (coords - coords[:, :1]).permute(0, 2, 1, 3).reshape(batch * tracks, frames, 3)
        flow = get_3d_embedding(flow, C=64)
        feature_tokens = features.permute(0, 2, 1, 3, 4).reshape(batch * tracks, frames, -1)
        tokens = torch.cat([flow, corr, feature_tokens], dim=-1) + position + temporal
        tokens = tokens.reshape(batch, tracks, frames, -1)
        update = module.update_former(tokens)
        if predict_radii:
            radii = module.control_former_(tokens)
        updated_features = []
        for plane in range(3):
            delta = update[..., 3 + 128 * plane:3 + 128 * (plane + 1)]
            delta = module.track_feat_updater(delta.reshape(batch * tracks * frames, 128))
            delta = delta.reshape(batch, tracks, frames, 128).permute(0, 2, 1, 3)
            updated_features.append(features[:, :, :, plane] + delta)
        features = torch.stack(updated_features, dim=-2)
        coords = coords + update[..., :3].permute(0, 2, 1, 3)
    return coords, radii
