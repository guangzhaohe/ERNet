"""Inference entry point for the two SceneTracker skin checkpoints."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='DT4D .anime file or template/points example directory')
    parser.add_argument('--mode', choices=['both', 'pcl', 'dep'], default=None)
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--weights-dir', type=Path, default=Path(__file__).parent / 'model')
    parser.add_argument('--no-preview', action='store_true', help='Skip MP4/PNG rendering')
    parser.add_argument('--render-only', action='store_true', help='Render existing saved predictions')
    parser.add_argument('--overwrite', action='store_true', help='Replace this case\'s existing outputs')
    args = parser.parse_args()
    selected = args.mode
    if selected is None and args.input.is_dir():
        selected = json.loads((args.input / 'demo.json').read_text())['mode']
    selected = selected or 'both'
    if selected not in ('pcl', 'dep', 'both'):
        parser.error(f'Invalid example mode: {selected}')
    modes = ('pcl', 'dep') if selected == 'both' else (selected,)
    out = args.output.resolve() / args.input.stem
    if not args.render_only:
        if not args.input.exists():
            parser.error(f'Input does not exist: {args.input}')
        if not args.overwrite and any((out / mode / 'predicted_vertices.npy').exists() for mode in modes):
            parser.error(f'Outputs exist in {out}; choose another --output or use --overwrite')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    out.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        from lib.inference import infer, infer_points
        run = infer_points if args.input.is_dir() else infer
        run(args.input.resolve(), out, modes, args.weights_dir.resolve())
    if not args.no_preview:
        from lib.visualization import render_previews
        render_previews(out, modes)
    print(f'Done: {out}')


if __name__ == '__main__':
    main()
