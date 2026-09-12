"""DT4D input, checkpoint-compatible sampling, normalization and exports."""
import json
import numpy as np


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def anime_read(filename):
    with open(filename, 'rb') as stream:
        header = np.fromfile(stream, dtype='<i4', count=3)
        if len(header) != 3 or np.any(header <= 0):
            raise ValueError(f'Invalid anime header: {filename}')
        frames, vertices, triangles = map(int, header)
        base = np.fromfile(stream, dtype='<f4', count=vertices * 3).reshape(vertices, 3)
        faces = np.fromfile(stream, dtype='<i4', count=triangles * 3).reshape(triangles, 3)
        offsets = np.fromfile(stream, dtype='<f4')
    if offsets.size != (frames - 1) * vertices * 3:
        raise ValueError(f'Invalid anime payload size: {filename}')
    if faces.min() < 0 or faces.max() >= vertices:
        raise ValueError('Face index outside the vertex array')
    if not np.isfinite(base).all() or not np.isfinite(offsets).all():
        raise ValueError('Nonfinite anime coordinates')
    return frames, vertices, triangles, base, faces, offsets.reshape(frames - 1, vertices, 3)


def normalize_sequence(vertices):
    """Match skin's per-frame centering and shared scale with padding 0.075."""
    centers = vertices.mean(axis=1, keepdims=True)
    normalized = vertices - centers
    extent = np.abs(normalized).max()
    if extent <= 0:
        raise ValueError('Cannot normalize a zero-size animation')
    ratio = (1. - 0.075) / extent
    normalized *= ratio
    transforms = np.eye(4)[None].repeat(len(vertices), axis=0)
    transforms[:, :3, :3] /= ratio
    transforms[:, 3, :3] = centers[:, 0]
    return normalized, transforms


def world_coords(points, transforms):
    return (np.einsum('fni,fij->fnj', points, transforms[:, :3, :3])
            + transforms[:, None, 3, :3]).astype(np.float32)


def reference_indices(points, count=8192):
    # Keep the original smart_sample(uni_probability=1) random-number sequence.
    np.random.uniform()
    indices = np.random.choice(len(points), min(count, len(points)), replace=False)
    np.random.shuffle(indices)
    return indices


def rand_uniform_sampler(vertices, triangles, num_samples):
    """The branch's surface sampler, including its historical barycentric rule."""
    tri = vertices[triangles]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    ids = np.random.choice(len(triangles), size=num_samples, p=areas / areas.sum())
    # Original code drew these before replacing them with normalized random weights.
    # Preserve RNG advancement so the reduced implementation reproduces prior runs.
    np.random.rand(num_samples, 1)
    np.random.rand(num_samples, 1)
    weights = np.random.uniform(0., 1., (num_samples, 3))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return (weights[:, :1] * tri[ids, 0] + weights[:, 1:2] * tri[ids, 1]
            + weights[:, 2:3] * tri[ids, 2])


def write_anime(path, vertices, faces):
    with path.open('wb') as stream:
        np.array([len(vertices), vertices.shape[1], len(faces)], '<i4').tofile(stream)
        vertices[0].astype('<f4').tofile(stream)
        faces.astype('<i4').tofile(stream)
        (vertices[1:] - vertices[0]).astype('<f4').tofile(stream)


def generate_random_semisphere_camera(radius=2, fov=60, size=300):
    theta = np.random.uniform(0, np.pi * 2)
    phi = np.random.uniform(0, np.pi / 2. - 1e-6)
    center = np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)]) * radius
    z = -center / np.linalg.norm(center)
    x = np.cross(np.array([0., 0., 1.]), center)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    rotation = np.stack([x, y, z]).reshape(3, 3)
    translation = (-rotation @ center.reshape(3, 1)).flatten()
    focal = size / (2. * np.tan(np.deg2rad(fov) / 2.))
    intrinsics = np.array([[focal, 0., size / 2], [0., focal, size / 2], [0., 0., 1.]])
    return intrinsics, rotation, translation


def back_project(k, r, t, depth):
    height, width = depth.shape
    grid = np.stack(np.meshgrid(np.arange(width), np.arange(height))).transpose(1, 2, 0)
    valid = depth > 1e-6
    pixels = grid[valid]
    homogeneous = np.ones((len(pixels), 3))
    homogeneous[:, :2] = pixels
    camera_points = depth[valid, None] * (homogeneous @ np.linalg.inv(k).T)
    return (camera_points - t) @ r


class Renderer:
    """Reusable EGL depth renderer with the branch's OpenCV/OpenGL convention."""
    def __init__(self, height=300, width=300):
        import pyrender
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(self, height, width, intrinsics, pose, mesh):
        import pyrender
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        camera = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                           fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(camera, pose=pose @ np.diag([1, -1, -1, 1]))
        return self.renderer.render(self.scene)

    @staticmethod
    def mesh_opengl(mesh):
        import pyrender
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
