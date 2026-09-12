"""The skin branch's trajectory-conditioned local rigid mesh blending."""
import torch
from knn_cuda import KNN

def knn(reference, query, count):
    return KNN(k=count, transpose_mode=True)(reference, query)

def no_nan(value):
    return bool(torch.isfinite(value).all())

def get_anchor_activation_mask(track_c, track_f, point_c, K, THRESH):
    B, F, T = track_f.shape[:3]
    N = point_c.shape[1]
    dist_c, idx_c = knn(track_c, point_c, K)
    order_c = torch.argsort(dist_c, dim=-1)
    idx_c = torch.gather(idx_c, dim=-1, index=order_c)
    dist_c = torch.gather(dist_c, dim=-1, index=order_c)
    node_c = track_c[0, idx_c.flatten(), :].reshape(N, K, 3)
    node_f = track_f[0, :, idx_c.flatten(), :].reshape(F, N, K, 3)
    node_dist_c = torch.norm(node_c - node_c[:, 0:1], dim=-1)
    node_dist_f = torch.norm(node_f - node_f[:, :, 0:1], dim=-1)
    node_dist_ratio = node_dist_f / node_dist_c[None]
    node_dist_ratio[:, :, 0] = 1.0
    node_dist_ratio_max = torch.max(node_dist_ratio, dim=0)[0] - 1.0
    node_dist_ratio_min = torch.min(node_dist_ratio, dim=0)[0] - 1.0
    node_dist_ratio_abs = torch.maximum(node_dist_ratio_max, torch.abs(node_dist_ratio_min))
    mask = node_dist_ratio_abs < THRESH
    return (idx_c, dist_c, node_dist_ratio_abs, mask)

def best_fit_transform_torch(A, B):
    assert A.shape == B.shape
    centroid_A = torch.mean(A, dim=1, keepdim=True)
    centroid_B = torch.mean(B, dim=1, keepdim=True)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.permute(0, 2, 1) @ BB
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.permute(0, 2, 1) @ U.permute(0, 2, 1)
    det = torch.linalg.det(R)
    Vt = Vt.clone()
    Vt[det < 0, -1, :] *= -1
    R = Vt.permute(0, 2, 1) @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    return (R, t)

def rotation_from_trajectories(track_c, track_f, K0: int, THRESH: float, K_min: int):
    B, F, T = track_f.shape[:3]
    idx_anchor, _, dist_anchor, mask_anchor = get_anchor_activation_mask(track_c, track_f, track_c, K0, THRESH)
    K = torch.min(torch.sum(mask_anchor, dim=-1))
    while K < K_min:
        THRESH += 0.1
        K0 = K0 + 1 if K0 < T else T
        idx_anchor, _, dist_anchor, mask_anchor = get_anchor_activation_mask(track_c, track_f, track_c, K0, THRESH)
        K = torch.min(torch.sum(mask_anchor, dim=-1))
    sort_anchor = torch.argsort(dist_anchor, dim=-1, descending=False)[:, :K]
    idx_anchor = torch.gather(idx_anchor[0], dim=-1, index=sort_anchor)
    local_c = track_c[:, idx_anchor.flatten(), :].reshape(B, T, K, 3)
    local_f = track_f[0, :, idx_anchor.flatten(), :].reshape(F, T, K, 3)
    local_c = local_c.repeat(F, 1, 1, 1)
    r, t = best_fit_transform_torch(local_c.reshape(F * T, K, 3), local_f.reshape(F * T, K, 3))
    r = r.reshape(F, T, 3, 3)
    t = t.reshape(F, T, 3)
    return (r, t)

def blend(point_c, track_c, track_f, radii_f, K: int, K_a: int, THRESH: float, K_min: int):
    B, N = point_c.shape[:2]
    F, T = track_f.shape[1:3]
    idx_c, dist_c, _, mask = get_anchor_activation_mask(track_c, track_f, point_c, K, THRESH)
    r = radii_f[0, :, idx_c.flatten(), 0].reshape(F, N, K)
    w_r = torch.exp(-dist_c ** 2 / (2 * r ** 2 + 1e-08))
    w = w_r.clone()
    w[:, ~mask] = 0.0
    w = w / torch.sum(w, dim=-1, keepdim=True)
    rot, tran = rotation_from_trajectories(track_c, track_f, K_a, THRESH, K_min)
    warp_rot = rot[:, idx_c.flatten()].reshape(F, N, K, 3, 3)
    warp_tran = tran[:, idx_c.flatten()].reshape(F, N, K, 3, 1)
    warp_raw = warp_rot @ point_c[:, :, None, :, None].repeat(F, 1, K, 1, 1) + warp_tran
    warp_raw = warp_raw[..., 0]
    warp_weighted = warp_raw * w[:, :, :, None]
    point_f = torch.sum(warp_weighted, dim=-2)
    assert no_nan(point_f)
    return point_f[None]

@torch.no_grad()
def eval_dqb(mesh_points, mesh_tracks, control_tracks, K, K_a, K_min, knn_thresh) -> torch.Tensor:
    return blend(mesh_points, mesh_tracks, control_tracks[..., :3], control_tracks[..., 3:], K, K_a, knn_thresh, K_min)
