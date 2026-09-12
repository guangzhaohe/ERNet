# ERNet bear template and point-sequence examples

Both examples come from the supplied `bear3EP_Agression.anime` case and contain:

- `template.ply`: the frame-0 template, 16,062 vertices and 31,124 triangles.
- `points.npz`: a float32 `points` array of shape `(101, 2048, 3)`.
- `demo.json`: model selection, file names, and preprocessing provenance.

```bash
conda activate scenetracker
python main.py assets/bear_pcl
python main.py assets/bear_dep
```

`bear_pcl` contains the uniform-surface samples from the earlier PCL run. These
are sparse point observations of the complete surface, not a camera-occluded
partial scan. `bear_dep` contains the fixed camera's visible depth points.
Keeping the original samples makes the demonstrations comparable to that run.

Templates and points are already in the model's normalized coordinate system
(approximately `[-1, 1]`). They were prepared with the original DT4D pipeline's
per-frame centering and shared scale, using the source animation's full meshes.
No source animation, ground-truth sequence, or additional calibration file is
needed at runtime. The examples demonstrate tracking a normalized sequence;
they do not recover world-space camera/object motion from raw scans.

For your own example, create a folder with the same three files. `demo.json`
needs `mode` (`pcl` or `dep`), `template` (mesh filename), and `points` (NPZ
filename). Put the template and all point frames in a common, model-normalized
coordinate system. The point count may differ, but must be constant across the
sequence. The template must contain at least 256 vertices.

Predicted meshes and controls are saved in the same normalized coordinates.
Previews label the static reference as **Reference template**, not ground truth.
Generated results belong in `output/`, not in `assets/`.
