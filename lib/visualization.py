"""Headless videos of templates, input points, predicted anchors and meshes."""
import colorsys
import numpy as np
import pyrender
import trimesh

def render_previews(out, modes=('pcl', 'dep')):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    template = trimesh.load(out / 'reference_mesh.ply', force='mesh', process=False)
    template_vertices = np.asarray(template.vertices)
    has_ground_truth = (out / 'ground_truth_vertices.npy').exists()
    if has_ground_truth:
        gt = np.load(out / 'ground_truth_vertices.npy')
    else:
        frames = len(np.load(out / modes[0] / 'input_points.npy', mmap_mode='r'))
        gt = np.broadcast_to(template_vertices, (frames, len(template_vertices), 3))
    faces = np.load(out / 'faces.npy')
    predictions = {m: np.load(out / m / 'predicted_vertices.npy') for m in modes}
    inputs = {m: np.load(out / m / 'input_points.npy') for m in modes}
    anchors = {m: np.load(out / m / 'predicted_controls.npy') for m in modes}
    sequences = [gt, template_vertices[None], *predictions.values(), *inputs.values(), *anchors.values()]
    low = np.min([a.min(axis=(0, 1)) for a in sequences], axis=0)
    high = np.max([a.max(axis=(0, 1)) for a in sequences], axis=0)
    center, scale = (low + high) / 2, 1.6 / (high - low).max()
    norm = lambda x: (x - center) * scale
    # A fixed, moderately elevated view keeps the complete sequence comparable.
    camera_center = np.array([2.6, -2.6, 1.8])
    z = -camera_center / np.linalg.norm(camera_center)
    x = np.cross([0, 0, 1], camera_center); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z])
    extrinsic = np.eye(4)
    extrinsic[:3, :3], extrinsic[:3, 3] = rotation, -rotation @ camera_center
    camera_pose = np.linalg.inv(extrinsic) @ np.diag([1, -1, -1, 1])
    scene = pyrender.Scene(bg_color=[0.96, 0.97, 0.98, 1], ambient_light=[0.55] * 3)
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(38)), pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(480, 480, point_size=3.0)
    def draw(points, title, color, mesh=True, point_size=3.0):
        renderer.point_size = point_size
        if mesh:
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(*color, 1),
                                                         metallicFactor=0, roughnessFactor=0.8)
            obj = pyrender.Mesh.from_trimesh(trimesh.Trimesh(norm(points), faces, process=False), material=material)
        else:
            colors = np.broadcast_to(color, (len(points), 3)).copy()
            obj = pyrender.Mesh.from_points(norm(points), colors=colors)
        node = scene.add(obj)
        rgb, _ = renderer.render(scene)
        scene.remove_node(node)
        panel = Image.new('RGB', (480, 512), 'white')
        panel.paste(Image.fromarray(rgb), (0, 32))
        ImageDraw.Draw(panel).text((12, 10), title, fill=(25, 30, 40))
        return np.asarray(panel)
    writers = {m: imageio.get_writer(out / m / 'preview.mp4', fps=24, codec='libx264', quality=8)
               for m in modes}
    comparison = imageio.get_writer(out / 'comparison.mp4', fps=24, codec='libx264', quality=8)
    try:
        template_panel = draw(template_vertices, 'Input template | reference frame 000', (0.30, 0.55, 0.82))
        anchor_colors = {m: np.array([colorsys.hsv_to_rgb(i / len(anchors[m][0]), 0.8, 0.8)
                                     for i in range(len(anchors[m][0]))]) for m in modes}
        for f in range(len(gt)):
            truth = draw(gt[f], f"{'Ground truth' if has_ground_truth else 'Reference template'} | frame {f:03d}", (0.30, 0.55, 0.82))
            pred = {}
            for mode in modes:
                color = {'pcl': (0.85, 0.48, 0.22), 'dep': (0.25, 0.65, 0.42)}[mode]
                pred[mode] = draw(predictions[mode][f], f'{mode.upper()} predicted mesh | frame {f:03d}', color)
                inp = draw(inputs[mode][f], f'Input points | {inputs[mode].shape[1]} points | frame {f:03d}', (0.25, 0.3, 0.4), mesh=False)
                anc = draw(anchors[mode][f], f'Predicted anchors | {anchors[mode].shape[1]} anchors | frame {f:03d}',
                           anchor_colors[mode], mesh=False, point_size=7.0)
                panels = np.concatenate([template_panel, inp, anc, pred[mode]], axis=1)
                writers[mode].append_data(panels)
                if f in [0, len(gt) // 2, len(gt) - 1]:
                    Image.fromarray(panels).save(out / mode / f'preview_{f:03d}.png')
            combined = np.concatenate([truth] + [pred[mode] for mode in modes], axis=1)
            comparison.append_data(combined)
            if f in [0, len(gt) // 2, len(gt) - 1]:
                Image.fromarray(combined).save(out / f'comparison_{f:03d}.png')
            if f % 20 == 0:
                print(f'Rendered previews {f + 1}/{len(gt)}', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        comparison.close()
        renderer.delete()
