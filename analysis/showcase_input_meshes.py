"""
Showcase the input meshes fed to the MANOGrasp GNNs, coloured by a chosen node
feature (default: the 7th feature, index 6 — the first Gaussian grasp/gaze weight).

Node feature layout (see grasp_mesh_set.PrecachedMANOGraspDataset.get):
    [ pos_x, pos_y, pos_z,  normal_x, normal_y, normal_z,  w_0, w_1, ... w_{P-1} ]
       0      1      2         3         4         5         6    7         ...
So feature index 6 (the 7th feature) is the first Gaussian weight column.

Each mesh is drawn as a wireframe: a faint translucent surface fill, grey edges
(from the cached edge_index), and feature-coloured node markers.

Two viewing modes:
  --mode scroll  (default)  one mesh at a time with a slider to step through them
  --mode grid               several meshes shown side-by-side at once

Both modes render with Plotly, whose toolbar includes a camera icon
("Download plot as PNG") to save the current view — configured here to export a
high-resolution PNG named after the mesh.

Examples:
    python showcase_input_meshes.py
    python showcase_input_meshes.py --mode grid --max 12
    python showcase_input_meshes.py --feature 6 --dataset .data_cache/data_src_val.pt
    python showcase_input_meshes.py --all-samples          # every sample, not just unique meshes
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from graspcnn.data import PrecachedMANOGraspDataset  # noqa: F401  (needed for torch.load)

# Where the original .glb meshes live (the paths stored in the dataset are
# relative and were created on another working dir).
DEFAULT_MESH_ROOT = '/home/hans-henrik-dalgaard/Documents/P10/hot3d'


# ── Geometry helpers ────────────────────────────────────────────────────────

def resolve_mesh_path(stored_path, mesh_root: str):
    """Find the .glb on disk: try the stored path, then mesh_root/stored_path,
    then search by basename under mesh_root."""
    stored = Path(stored_path)
    if stored.exists():
        return stored
    candidate = Path(mesh_root) / stored
    if candidate.exists():
        return candidate
    hits = list(Path(mesh_root).rglob(stored.name))
    return hits[0] if hits else None


def load_faces(mesh_meta, mesh_root: str):
    """Return (F, 3) face index array for a mesh, or None if unavailable."""
    path = resolve_mesh_path(mesh_meta.path, mesh_root)
    if path is None:
        print(f'[warn] mesh file not found for uid={mesh_meta.uid} '
              f'({mesh_meta.path}); falling back to point cloud.')
        return None
    try:
        tm = trimesh.load(str(path), force='mesh')
        return tm.faces
    except Exception as exc:  # noqa: BLE001
        print(f'[warn] could not load faces for {path}: {exc}; using point cloud.')
        return None


def collect_entries(dataset, feature_idx: int, mesh_root: str, all_samples: bool):
    """Build a list of dicts holding everything needed to render each mesh.

    By default keeps one sample per unique mesh uid (so we 'showcase the input
    meshes'); --all-samples keeps every sample (the feature differs per sample).
    """
    entries = []
    seen = set()
    proc_dir = Path(dataset.processed_dir)
    mesh_by_uid = {m.uid: m for m in dataset.meshes}

    for idx in range(len(dataset)):
        sample_hash = dataset.sample_hashes[idx]
        sample_raw = torch.load(proc_dir / f'sample_{sample_hash}.pt', weights_only=False)
        uid = sample_raw['uid']
        if not all_samples and uid in seen:
            continue
        seen.add(uid)

        data = dataset[idx]
        if feature_idx >= data.x.shape[1]:
            raise IndexError(f'feature index {feature_idx} out of range; '
                             f'mesh has {data.x.shape[1]} node features.')
        feat = data.x[:, feature_idx].numpy()

        mesh_meta = mesh_by_uid.get(uid)
        mesh_raw = torch.load(proc_dir / f'mesh_{uid}.pt', weights_only=False)
        pos = mesh_raw['pos'].numpy()
        edge_index = mesh_raw['edge_index'].numpy()
        faces = load_faces(mesh_meta, mesh_root) if mesh_meta is not None else None

        entries.append(dict(uid=uid, idx=idx, pos=pos, faces=faces,
                            edge_index=edge_index, feat=feat))
        label = f'{len(entries):>3}: uid={uid}  sample#{idx}  ' \
                f'feat[{feature_idx}] range [{feat.min():.3f}, {feat.max():.3f}]'
        print(label)

    return entries


# ── Plotly trace builders ───────────────────────────────────────────────────

_AXIS = dict(showticklabels=False, showgrid=False, zeroline=False,
             showbackground=False, title='')
_SCENE = dict(xaxis=_AXIS, yaxis=_AXIS, zaxis=_AXIS, aspectmode='data',
              camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)))


def edge_segments(pos, edge_index):
    """Build (xs, ys, zs) line-segment arrays with None separators between edges.
    edge_index is (2, E) and undirected (both directions present) — keep i<j to
    draw each edge once."""
    src, dst = edge_index[0], edge_index[1]
    keep = src < dst
    src, dst = src[keep], dst[keep]
    n = src.shape[0]
    xs = np.empty(n * 3); ys = np.empty(n * 3); zs = np.empty(n * 3)
    for axis, out in enumerate((xs, ys, zs)):
        out[0::3] = pos[src, axis]
        out[1::3] = pos[dst, axis]
        out[2::3] = np.nan  # break between segments
    return xs, ys, zs


def make_traces(entry, showscale, colorbar_title, fill_opacity=0.15):
    """Return a list of traces rendering the mesh as a wireframe: a faint
    translucent fill, grey edges, and feature-coloured node markers."""
    pos, faces, feat = entry['pos'], entry['faces'], entry['feat']
    traces = []

    # 1. Light low-opacity surface fill (only when faces are available).
    if faces is not None and fill_opacity > 0:
        traces.append(go.Mesh3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='lightsteelblue', opacity=fill_opacity, flatshading=True,
            hoverinfo='skip', showscale=False, name=f'{entry["uid"]} fill',
        ))

    # 2. Edges as thin grey lines.
    xs, ys, zs = edge_segments(pos, entry['edge_index'])
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(color='rgba(90,90,90,0.65)', width=1.2),
        hoverinfo='skip', showlegend=False, name=f'{entry["uid"]} edges',
    ))

    # 3. Nodes coloured by the chosen feature (carries the colour bar).
    traces.append(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='markers',
        marker=dict(size=3, color=feat, colorscale='Viridis',
                    cmin=feat.min(), cmax=feat.max(), showscale=showscale,
                    colorbar=dict(title=colorbar_title, thickness=14) if showscale else None),
        hoverinfo='skip', showlegend=False, name=f'{entry["uid"]} nodes',
    ))
    return traces


def modebar_config(name: str):
    """Plotly config that makes the toolbar PNG-download button export a clean,
    high-resolution image named after the view."""
    return dict(
        displaylogo=False,
        modeBarButtonsToAdd=['toImage'],
        toImageButtonOptions=dict(format='png', filename=name, scale=3),
    )


# ── Scroll mode ─────────────────────────────────────────────────────────────

def show_scroll(entries, feature_idx, fill_opacity):
    cbar = f'gaze_weight'
    fig = go.Figure()

    # Each mesh contributes several traces (fill/edges/nodes); only the first
    # mesh's traces start visible. The slider toggles whole groups so the
    # toolbar's PNG button always captures exactly the current mesh.
    trace_owner = []  # which mesh index each added trace belongs to
    for i, entry in enumerate(entries):
        for tr in make_traces(entry, showscale=True, colorbar_title=cbar,
                               fill_opacity=fill_opacity):
            tr.visible = (i == 0)
            fig.add_trace(tr)
            trace_owner.append(i)

    steps = []
    for i, entry in enumerate(entries):
        vis = [owner == i for owner in trace_owner]
        steps.append(dict(
            method='update',
            args=[{'visible': vis},
                  {'title.text': f'Input mesh {i + 1}/{len(entries)} — '
                                 f'uid={entry["uid"]} (sample #{entry["idx"]}) — '
                                 f'coloured by node gaze weight'}],
            label=str(i + 1),
        ))

    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix='Mesh: '), steps=steps,
                      pad=dict(t=30))],
        scene=_SCENE,
        title=dict(text=f'Input mesh 1/{len(entries)} — uid={entries[0]["uid"]} '
                        f'(sample #{entries[0]["idx"]}) — '
                        f'coloured by node feature[{feature_idx}]', x=0.5),
        height=720, paper_bgcolor='white',
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.show(config=modebar_config(f'input_mesh_feature{feature_idx}'))


# ── Grid mode ───────────────────────────────────────────────────────────────

def show_grid(entries, feature_idx, max_n, fill_opacity):
    entries = entries[:max_n]
    n = len(entries)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    cbar = f'feature[{feature_idx}]'

    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    titles = [f'uid={e["uid"]}  (s#{e["idx"]})' for e in entries]
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles,
                        horizontal_spacing=0.01, vertical_spacing=0.05)

    for k, entry in enumerate(entries):
        r, c = divmod(k, cols)
        # Show one shared colour bar (last panel) to keep the layout clean.
        for tr in make_traces(entry, showscale=(k == n - 1), colorbar_title=cbar,
                              fill_opacity=fill_opacity):
            fig.add_trace(tr, row=r + 1, col=c + 1)
        scene_key = 'scene' if k == 0 else f'scene{k + 1}'
        fig.update_layout(**{scene_key: _SCENE})

    fig.update_layout(
        title=dict(text=f'Input meshes for the GNN — coloured by node '
                        f'feature[{feature_idx}]  ({n} meshes)', x=0.5),
        height=360 * rows, paper_bgcolor='white',
        margin=dict(l=0, r=0, t=70, b=0),
    )
    fig.show(config=modebar_config(f'input_meshes_grid_feature{feature_idx}'))


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Showcase GNN input meshes coloured by a node feature.')
    parser.add_argument('--dataset', default='.data_cache/data_src_train.pt',
                        help='Serialised PrecachedMANOGraspDataset (.pt).')
    parser.add_argument('--feature', type=int, default=6,
                        help='Node-feature column to colour by (default 6 = the '
                             '7th feature, first Gaussian weight).')
    parser.add_argument('--mode', choices=['scroll', 'grid'], default='scroll',
                        help='scroll: slider through meshes; grid: many at once.')
    parser.add_argument('--max', type=int, default=12,
                        help='Max meshes to show in grid mode (default 12).')
    parser.add_argument('--all-samples', action='store_true',
                        help='Use every sample instead of one per unique mesh.')
    parser.add_argument('--fill-opacity', type=float, default=0.15,
                        help='Opacity of the faint surface fill behind the '
                             'wireframe (0 disables it; default 0.15).')
    parser.add_argument('--mesh-root', default=DEFAULT_MESH_ROOT,
                        help='Root dir to resolve the original .glb mesh files.')
    args = parser.parse_args()

    print(f'Loading dataset: {args.dataset}')
    dataset = torch.load(args.dataset, weights_only=False)
    print(f'{len(dataset)} samples, {len(dataset.meshes)} unique meshes. '
          f'Colouring by node feature[{args.feature}].\n')

    entries = collect_entries(dataset, args.feature, args.mesh_root, args.all_samples)
    if not entries:
        print('No meshes to show.')
        return
    print(f'\nRendering {len(entries)} mesh(es) in {args.mode} mode.')

    if args.mode == 'scroll':
        show_scroll(entries, args.feature, args.fill_opacity)
    else:
        show_grid(entries, args.feature, args.max, args.fill_opacity)


if __name__ == '__main__':
    main()
