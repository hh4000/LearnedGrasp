"""
Visualise information propagation through MANOGraspGNN models.

Three interactive Plotly figures:
  1. Spatial   — 3D mesh coloured by per-node activation magnitude at each layer
  2. Embedding — 2D PCA of node embeddings per GCN layer, coloured by gaze weight
  3. Stats     — violin plots of activation magnitude per layer

Model class and n_out are detected automatically from the checkpoint.
Traceback (most recent call last):
  File "/home/hans-henrik-dalgaard/Documents/P10/GraspCNN/visualise_gnn.py", line 280, in <module>
    visualise(
    ~~~~~~~~~^
        dataset_path=args.dataset,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        model_path=args.model,
        ^^^^^^^^^^^^^^^^^^^^^^
        sample_idx=args.sample,
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/hans-henrik-dalgaard/Documents/P10/GraspCNN/visualise_gnn.py", line 181, in visualise
    model, model_name, n_out = detect_model(model_path)
                               ~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/hans-henrik-dalgaard/Documents/P10/GraspCNN/visualise_gnn.py", line 105, in detect_model
    model.load_state_dict(state)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/hans-henrik-dalgaard/miniconda3/envs/graspcnn/lib/python3.13/site-packages/torch/nn/modules/module.py", line 2639, in load_state_dict
    raise RuntimeError(
    ...<3 lines>...
    )
RuntimeError: Error(s) in loading state_dict for MANOGraspGNNv1:
	Unexpected key(s) in state_dict: "gnns.3.bias", "gnns.3.lin.weight", "gnns.4.bias", "gnns.4.lin.weight", "mlp.9.weight", "mlp.9.bias".
	size mismatch for gnns.1.lin.weight: copying a param with shape torch.Size([64, 75]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for gnns.2.lin.weight: copying a param with shape torch.Size([64, 139]) from checkpoint, the shape in current model is torch.Size([64, 64]).
	size mismatch for mlp.0.weight: copying a param with shape torch.Size([256, 998]) from checkpoint, the shape in current model is torch.Size([128, 133]).
	size mismatch for mlp.0.bias: copying a param with shape torch.Size([256]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for mlp.3.weight: copying a param with shape torch.Size([128, 256]) from checkpoint, the shape in current model is torch.Size([64, 128]).
	size mismatch for mlp.3.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for mlp.6.weight: copying a param with shape torch.Size([64, 128]) from checkpoint, the shape in current model is torch.Size([64, 64]).
Usage:
    python visualise_gnn.py --model models/XXXXXX.pth --dataset data/train_gnn.pt
    python visualise_gnn.py --model models/XXXXXX.pth --dataset data/train_gnn.pt --sample 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.data import Batch

from MODELS import *
from grasp_mesh_set import PrecachedMANOGraspDataset

MODELS = {
    'MANOGraspGNNv1': MANOGraspGNNv1,
    'MANOGraspGNNv2': MANOGraspGNNv2,
    'MANOGraspGNNv3': MANOGraspGNNv3,
    'MANOGraspGNNv4': MANOGraspGNNv4,
    'MANOGraspGNNv5': MANOGraspGNNv5,
    'MANOGraspGNNv6': MANOGraspGNNv6,
    'MANOGraspGNNv7': MANOGraspGNNv7,
}


# ── Geometry loading ──────────────────────────────────────────────────────────

def load_geometry(dataset: PrecachedMANOGraspDataset, sample_idx: int):
    """Return (pos, faces_or_None, gaze_weights, PyG Data) for one sample."""
    proc_dir = Path(dataset.processed_dir)
    sample_hash = dataset.sample_hashes[sample_idx]

    sample_raw = torch.load(proc_dir / f'sample_{sample_hash}.pt', weights_only=False)
    uid        = sample_raw['uid']
    mesh_raw   = torch.load(proc_dir / f'mesh_{uid}.pt', weights_only=False)

    pos  = mesh_raw['pos'].numpy()                   # (N, 3)
    gaze = sample_raw['weights'].norm(dim=1).numpy() # (N,)
    gaze /= gaze.max()

    faces = None
    mesh_meta = next((m for m in dataset.meshes if m.uid == uid), None)
    if mesh_meta is not None:
        try:
            tm = trimesh.load(str(mesh_meta.path), force='mesh')
            faces = tm.faces  # (F, 3)
        except Exception as exc:
            print(f'[warn] Could not load mesh faces ({exc}); using point cloud.')

    pyg_sample = dataset[sample_idx]
    return pos, faces, gaze, pyg_sample


# ── Model auto-detection ──────────────────────────────────────────────────────

def detect_model(model_path: str):
    """Load a checkpoint and return (model, model_name, n_out), auto-detecting both."""
    path  = Path(model_path)
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    state = {k.removeprefix('module.'): v for k, v in state.items()}

    # n_out: output size of the final MLP linear layer
    n_in = state['gnns.0.lin.weight'].shape[1]
    mlp_weight_keys = sorted(k for k in state if k.startswith('mlp.') and k.endswith('.weight'))
    n_out = state[mlp_weight_keys[-1]].shape[0]

    # Primary: params JSON written by the training script (same run_id as .pth stem)
    model_name = None
    params_path = path.parent.parent / 'params' / f'params_{path.stem}.json'
    if params_path.exists():
        params = json.load(open(params_path))
        first_line = params.get('architecture', '').strip().split('\n')[0]
        candidate  = first_line.rstrip('(').strip()
        if candidate in MODELS:
            model_name = candidate

    # Fallback: infer from state dict structure
    # EdgeConv keys contain 'nn.0.weight'; GCNConv keys contain 'lin.weight'
    # MLP input width: 133 = 2×64+5 (v1/v2), 197 = 3×64+5 (v3/v4)
    # n_gnn_layers distinguishes v7 (5 layers with skip-connections) from v1/v4 (3 layers)
    if model_name is None:
        has_edge_conv = any(k.startswith('gnns') and 'nn.0.weight' in k for k in state)
        mlp_in = state['mlp.0.weight'].shape[1]
        n_gnn_layers = sum(
            1 for k in state
            if k.startswith('gnns.') and k.endswith('.lin.weight') and 'nn.' not in k
        )
        model_name = {
            (False, 133, 3): 'MANOGraspGNNv1',
            (True,  133, 3): 'MANOGraspGNNv2',
            (True,  197, 3): 'MANOGraspGNNv3',
            (False, 197, 3): 'MANOGraspGNNv4',
            (False, mlp_in, 5): 'MANOGraspGNNv7',
        }.get((has_edge_conv, mlp_in, n_gnn_layers), 'MANOGraspGNNv1')

    model = MODELS[model_name](dropoutgnn=0.0, dropoutfc=0.0, n_in=n_in, n_out=n_out)
    model.load_state_dict(state)
    model.eval()
    print(f'Detected: {model_name}  |  n_out={n_out}')
    return model, model_name, n_out


# ── Hook registration ─────────────────────────────────────────────────────────

def register_hooks(model):
    """Capture post-regularisation node embeddings after each GNN layer."""
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for i, reg in enumerate(model.regs):
        def _hook(_mod, _inp, out, _i=i):
            activations[f'Layer {_i + 1}'] = out.detach().cpu()
        handles.append(reg.register_forward_hook(_hook))
    return activations, handles


# ── PCA (no sklearn needed) ───────────────────────────────────────────────────

def pca2d(embeddings: np.ndarray):
    """Project N×D to N×2 via randomised PCA.  Returns (coords N×2, [r1, r2])."""
    x   = torch.tensor(embeddings, dtype=torch.float)
    x_c = x - x.mean(0)
    _, S, V = torch.pca_lowrank(x_c, q=min(2, x_c.shape[1]), niter=4)
    coords = (x_c @ V).numpy()
    total  = (x_c ** 2).sum().item() + 1e-12
    ratios = ((S ** 2) / total).tolist()
    return coords, ratios


# ── Colour normalisation ──────────────────────────────────────────────────────

def norm01(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-12)


# ── Plotly trace helpers ──────────────────────────────────────────────────────

_AXIS = dict(showticklabels=False, showgrid=False, zeroline=False,
             showbackground=False, title='')
_SCENE = dict(xaxis=_AXIS, yaxis=_AXIS, zaxis=_AXIS, aspectmode='data',
              camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)))


def mesh3d_trace(pos, faces, intensity, name, showscale=False):
    return go.Mesh3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=intensity, colorscale='Viridis',
        showscale=showscale, reversescale=False,
        name=name, hoverinfo='skip',
        lighting=dict(ambient=0.5, diffuse=0.9, roughness=0.5),
    )


def scatter3d_trace(pos, intensity, name, showscale=False):
    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(size=2, color=intensity, colorscale='Viridis',
                    showscale=showscale),
        name=name, hoverinfo='skip',
    )


# ── Main visualisation ────────────────────────────────────────────────────────

def visualise(dataset_path: str, model_path: str, sample_idx: int = 0):

    # ── Load data & model ─────────────────────────────────────────────────────
    dataset: PrecachedMANOGraspDataset = torch.load(dataset_path, weights_only=False)
    pos, faces, gaze, pyg_sample = load_geometry(dataset, sample_idx)

    model, model_name, n_out = detect_model(model_path)

    activations, handles = register_hooks(model)
    with torch.no_grad():
        batch = Batch.from_data_list([pyg_sample])
        pred  = model(batch)
    for h in handles:
        h.remove()

    print(f'Prediction: {pred.squeeze().tolist()}')
    print(f'Label:      {pyg_sample.y.squeeze()[:n_out].tolist()}')

    # ── Layer data ────────────────────────────────────────────────────────────
    # Colour each node by ||embedding_i - embedding_{gaze_node}||_2
    gaze_node_idx = int(np.argmax(gaze))

    x_np = pyg_sample.x.numpy()                                    # (N, F)
    input_dist = x_np[:,-1]  # (N,)

    layer_scalars = {'Input\n(features)': input_dist}
    layer_embeds  = {}
    for name, act in activations.items():
        act_np = act.numpy()                                        # (N, D)
        dist   = np.linalg.norm(act_np - act_np[gaze_node_idx], axis=1)  # (N,)
        layer_scalars[name] = dist
        layer_embeds[name]  = act_np

    # ── Figure 1: Spatial propagation ─────────────────────────────────────────
    titles  = list(layer_scalars.keys())
    n_panel = len(titles)
    specs   = [[{'type': 'mesh3d' if faces is not None else 'scatter3d'}] * n_panel]

    fig1 = make_subplots(rows=1, cols=n_panel, specs=specs,
                         subplot_titles=titles, horizontal_spacing=0.01)

    for col, (title, scalars) in enumerate(layer_scalars.items(), start=1):
        intensity = norm01(scalars)
        trace = (mesh3d_trace(pos, faces, intensity, title, showscale=(col == n_panel))
                 if faces is not None
                 else scatter3d_trace(pos, intensity, title, showscale=(col == n_panel)))
        fig1.add_trace(trace, row=1, col=col)
        scene_key = 'scene' if col == 1 else f'scene{col}'
        fig1.update_layout(**{scene_key: _SCENE})

    fig1.update_layout(
        title=dict(text='Spatial propagation — L2 distance from highest-gaze node per layer', x=0.5),
        height=520, paper_bgcolor='white',
    )

    # ── Figure 2: Embedding PCA ────────────────────────────────────────────────
    n_embed = len(layer_embeds)
    fig2 = make_subplots(rows=1, cols=n_embed,
                         subplot_titles=list(layer_embeds.keys()),
                         horizontal_spacing=0.10)

    gaze_norm = norm01(gaze)
    for col, (name, emb) in enumerate(layer_embeds.items(), start=1):
        coords, ratios = pca2d(emb)
        fig2.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode='markers',
            marker=dict(size=3, color=gaze_norm, colorscale='Plasma',
                        showscale=(col == n_embed),
                        colorbar=dict(title='Gaze', thickness=12)),
            name=name,
            hovertemplate='PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>',
        ), row=1, col=col)
        fig2.update_xaxes(title_text=f'PC1 ({ratios[0]:.1%} var)', row=1, col=col)
        fig2.update_yaxes(title_text='PC2' if col == 1 else '', row=1, col=col)

    fig2.update_layout(
        title=dict(text='Node embedding PCA per layer — colour = gaze weight', x=0.5),
        height=420, showlegend=False, paper_bgcolor='white',
    )

    # ── Figure 3: Feature norm distributions ──────────────────────────────────
    fig3 = go.Figure()
    for name, scalars in layer_scalars.items():
        fig3.add_trace(go.Violin(
            y=scalars, name=name.replace('\n', ' '),
            box_visible=True, meanline_visible=True, points=False,
        ))

    fig3.update_layout(
        title=dict(text='Per-node L2 distance from highest-gaze node per layer', x=0.5),
        yaxis_title='L2 distance from gaze node',
        height=380, paper_bgcolor='white',
    )

    fig1.show()
    fig2.show()
    fig3.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN layer propagation visualiser')
    parser.add_argument('--model',   required=True,
                        help='Path to model state dict (.pth)')
    parser.add_argument('--dataset', required=True,
                        help='Path to serialised PrecachedMANOGraspDataset (.pt)')
    parser.add_argument('--sample',  type=int, default=0,
                        help='Sample index to visualise (default: 0)')
    args = parser.parse_args()

    visualise(
        dataset_path=args.dataset,
        model_path=args.model,
        sample_idx=args.sample,
    )
