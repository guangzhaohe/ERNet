"""PCL/DEP inference from DT4D animations or template-plus-points examples."""
import json
import time
import numpy as np
import torch
import fpsample
import trimesh
from .data import (anime_read, normalize_sequence, reference_indices, world_coords,
                   write_anime, save_json, rand_uniform_sampler, Renderer,
                   back_project, generate_random_semisphere_camera)
from .network.scene_tracker import SceneTracker
from .geometry import eval_dqb

CHECKPOINTS = {'pcl': 'train_w_skinning_647500.pth',
               'dep': 'train_w_skinning_dep_542200.pth'}


def _reference(template, faces, out):
    if len(template) < 256 or not np.isfinite(template).all():
        raise ValueError('A finite template with at least 256 vertices is required')
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    vertex_ids = reference_indices(template)
    reference = template[vertex_ids].astype(np.float32)
    anchor_ids = fpsample.bucket_fps_kdline_sampling(reference, 256, h=5, start_idx=0)
    np.save(out / 'faces.npy', faces)
    np.save(out / 'reference_vertex_indices.npy', vertex_ids)
    np.save(out / 'control_vertex_indices.npy', vertex_ids[anchor_ids])
    return reference, reference[anchor_ids]


def _sample_sequence(mode, vertices, faces, folder):
    np.random.seed(42)
    point_frames, depth_maps = [], []
    renderer = None
    if mode == 'dep':
        k, r, t = generate_random_semisphere_camera(radius=2, fov=60, size=300)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r, t
        renderer = Renderer(300, 300)
        np.savez(folder / 'camera.npz', intrinsics=k, world_to_camera_rotation=r,
                 world_to_camera_translation=t, camera_to_world=np.linalg.inv(pose))
    try:
        for frame, mesh_vertices in enumerate(vertices):
            if mode == 'pcl':
                dense = rand_uniform_sampler(mesh_vertices, faces, 3 * 2048)
            else:
                mesh = renderer.mesh_opengl(trimesh.Trimesh(mesh_vertices, faces, process=False))
                _, depth = renderer(300, 300, k, np.linalg.inv(pose), mesh)
                dense = back_project(k, r, t, depth)
                depth_maps.append(depth.astype(np.float32))
                if len(dense) < 2048:
                    raise ValueError(f'Frame {frame}: only {len(dense)} visible depth points')
            ids = fpsample.bucket_fps_kdline_sampling(dense, 2048, h=5, start_idx=0)
            point_frames.append(dense[ids].astype(np.float32))
            if frame % 20 == 0 or frame == len(vertices) - 1:
                print(f'{mode}: sampled {frame + 1}/{len(vertices)} frames', flush=True)
    finally:
        if renderer:
            renderer.delete()
    if depth_maps:
        np.save(folder / 'depth_maps.npy', np.stack(depth_maps))
    return np.clip(np.stack(point_frames), -1 + 1e-5, 1 - 1e-5)


def _predict(points, template, faces, reference, anchors, checkpoint, folder,
             transforms, ground_truth=None):
    mode = folder.name
    started = time.perf_counter()
    frames = len(points)
    np.save(folder / 'input_points.npy', world_coords(points, transforms))
    model = SceneTracker.from_checkpoint(checkpoint)
    print(f'{mode}: strictly loaded {len(model.state_dict())} state entries', flush=True)
    def tensor(array):
        return torch.as_tensor(array, dtype=torch.float32, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        controls = model(tensor(points)[None], tensor(reference)[None], tensor(anchors)[None],
                         progress=lambda msg: print(f'{mode}: {msg}', flush=True))
        if controls.shape != (1, frames, 256, 4) or not torch.isfinite(controls).all():
            raise RuntimeError('Invalid control predictions')
        print(f'{mode}: warping all {len(template)} mesh vertices', flush=True)
        prediction = eval_dqb(tensor(template)[None], tensor(anchors)[None], controls, 4, 8, 4, 0.2)[0]
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite mesh predictions')
        pred_norm = prediction.cpu().numpy()
        controls_np = controls[0].cpu().numpy()
    predicted = world_coords(pred_norm, transforms)
    np.save(folder / 'predicted_vertices.npy', predicted)
    np.save(folder / 'predicted_controls.npy', world_coords(controls_np[..., :3], transforms))
    np.save(folder / 'control_radii_normalized.npy', controls_np[..., 3])
    write_anime(folder / 'prediction.anime', predicted, faces)
    metrics = dict(checkpoint=str(checkpoint), frames=frames, vertices=len(template),
                   elapsed_seconds=time.perf_counter() - started,
                   peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
    if ground_truth is not None:
        errors = np.linalg.norm(predicted - ground_truth, axis=-1)
        np.save(folder / 'vertex_errors.npy', errors)
        diagonal = float(np.linalg.norm(np.ptp(ground_truth[0], axis=0)))
        metrics.update(mean_vertex_error=float(errors.mean()),
                       mean_error_percent_reference_bbox_diagonal=float(errors.mean() / diagonal * 100),
                       per_frame_mean_vertex_error=errors.mean(axis=1).tolist())
    save_json(folder / 'metrics.json', metrics)
    print(f'{mode}: saved {frames} predicted frames', flush=True)
    del model, controls, prediction
    torch.cuda.empty_cache()
    return metrics


def infer(anime, out, modes, weights_dir):
    """DT4D evaluation, preserving the original branch's mesh-based centering."""
    nf, nv, nt, vertices, faces, offsets = anime_read(str(anime))
    ground_truth = np.concatenate([vertices[None], vertices + offsets]).astype(np.float32)
    normalized, transforms = normalize_sequence(ground_truth)
    reference, anchors = _reference(normalized[0], faces, out)
    np.save(out / 'ground_truth_vertices.npy', ground_truth)
    np.save(out / 'reverse_transforms.npy', transforms)
    trimesh.Trimesh(vertices, faces, process=False).export(out / 'reference_mesh.ply')
    print(f'Loaded {nf} frames, {nv} vertices, {nt} triangles', flush=True)
    save_json(out / 'metadata.json', dict(
        source=str(anime), frames=nf, vertices=nv, triangles=nt, seed=42,
        reference_frame=0, reference_points=len(reference), controls=256,
        input_points=2048, window=8, update_iterations=6,
        coordinate_system='Original anime coordinates; control radii are model-normalized.',
        preprocessing='Per-frame full-mesh centroid and shared scale, padding 0.075, no augmentation. This evaluation uses source meshes for normalization, not raw-depth-only world-pose estimation.'))
    results = {}
    for mode in modes:
        folder = out / mode
        folder.mkdir(exist_ok=True)
        points = _sample_sequence(mode, normalized, faces, folder)
        results[mode] = _predict(points, normalized[0], faces, reference, anchors,
                                 weights_dir / CHECKPOINTS[mode], folder, transforms, ground_truth)
    save_json(out / 'comparison.json', results)


def infer_points(example, out, modes, weights_dir):
    """Infer directly from an already normalized template and point sequence."""
    metadata = json.loads((example / 'demo.json').read_text())
    mesh = trimesh.load(example / metadata['template'], force='mesh', process=False)
    template, faces = np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)
    with np.load(example / metadata['points'], allow_pickle=False) as archive:
        points = archive['points'].astype(np.float32)
    if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
        raise ValueError('points.npz must contain points with shape [frames, points, 3]')
    if not np.isfinite(points).all() or max(np.abs(points).max(), np.abs(template).max()) > 1.01:
        raise ValueError('Template and points must be finite and pre-normalized to [-1, 1]')
    reference, anchors = _reference(template, faces, out)
    mesh.export(out / 'reference_mesh.ply')
    transforms = np.eye(4)[None].repeat(len(points), axis=0)
    save_json(out / 'metadata.json', dict(source=str(example), modes=list(modes),
              frames=len(points), input_points=points.shape[1], controls=256,
              coordinate_system='Model-normalized, matching the supplied template and points.',
              ground_truth_used=False, input_metadata=metadata))
    results = {}
    for mode in modes:
        folder = out / mode
        folder.mkdir(exist_ok=True)
        results[mode] = _predict(points, template, faces, reference, anchors,
                                 weights_dir / CHECKPOINTS[mode], folder, transforms)
    save_json(out / 'comparison.json', results)
