"""NiceGUI layer-output visualiser for GraspCNN models."""

import asyncio
import base64
import io
import json
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from nicegui import ui

from MODELS import *
from grasp_image_set import GraspImageDataset

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'params')
DATA_DIR   = os.path.expanduser('~/Documents/P10/hot3d/data')

MODELS = {
    'GraspCNNv1': GraspCNNv1,
    'GraspCNNv2': GraspCNNv2,
    'GraspCNNv3': GraspCNNv3,
    'MANOGraspCNNv1': MANOGraspCNNv1,
    'MANOGraspCNNv2': MANOGraspCNNv2,
    'MANOGraspCNNv3': MANOGraspCNNv3,
}
CLASS_NAMES = ['Lateral', 'Pinch', 'Power']
DATASETS = {
    'Train': os.path.join(DATA_DIR, 'train_images_poses.pt'),
    'Val':   os.path.join(DATA_DIR, 'val_images_poses.pt'),
    'Test':  os.path.join(DATA_DIR, 'test_images_poses.pt'),
}

# ── Rendering helpers ─────────────────────────────────────────────────────────

def _fig_to_src(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def render_input(image: torch.Tensor) -> str:
    """image: (4, H, W) — channels R, G, B, Gaze (may be float16)."""
    arr = image.float().numpy()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))
    for i, (ax, title) in enumerate(zip(axes, ['R', 'G', 'B', 'Gaze', 'RGB'])):
        if i < 4:
            ax.imshow(_norm01(arr[i]), cmap='gray' if i == 3 else 'viridis', vmin=0, vmax=1)
        else:
            ax.imshow(np.stack([_norm01(arr[c]) for c in range(3)], axis=-1))
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle('Input', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return _fig_to_src(fig)


def render_feature_maps(tensor: torch.Tensor, title: str, max_cols: int = 16) -> str:
    """tensor: (C, H, W) — fixed 10-inch-wide grid, height scales with rows."""
    arr = tensor.float().numpy()
    C, H, W = arr.shape
    n_cols = min(C, max_cols)
    n_rows = (C + n_cols - 1) // n_cols
    # Fixed total width; cell height preserves H:W but is capped to keep rows compact
    fig_w  = 10.0
    cell_w = fig_w / n_cols
    cell_h = min(cell_w * (H / max(W, 1)), 1.2)   # cap so tall maps don't blow out height
    cell_h = max(cell_h, 0.35)                      # floor so 1×1 maps still show
    fig_h  = n_rows * cell_h + 0.5                  # 0.5 for title
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    vmin, vmax = arr.min(), arr.max()
    for r in range(n_rows):
        for c in range(n_cols):
            ax  = axes[r][c]
            idx = r * n_cols + c
            if idx < C:
                ax.imshow(arr[idx], cmap='viridis', vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='auto')
            ax.axis('off')
    fig.suptitle(title, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.1)
    return _fig_to_src(fig) 


def render_vector(tensor: torch.Tensor, title: str, class_labels: list[str] | None = None) -> str:
    """tensor: (N,) — bar chart for small N, heatmap for large N. Fixed width."""
    arr = tensor.float().numpy()
    N   = len(arr)

    if N <= 256:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.bar(np.arange(N), arr, color='#4c8bf5', edgecolor='none', width=0.9)
        ax.axhline(0, color='black', linewidth=0.5)
        if class_labels and len(class_labels) == N:
            ax.set_xticks(np.arange(N))
            ax.set_xticklabels(class_labels, rotation=20, fontsize=8)
        else:
            ax.set_xlabel('Feature index', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.tick_params(labelsize=7)
    else:
        # Reshape into a fixed-width heatmap
        n_cols = 64
        n_rows = (N + n_cols - 1) // n_cols
        padded = np.pad(arr, (0, n_rows * n_cols - N))
        grid   = padded.reshape(n_rows, n_cols)
        row_h  = min(0.3, 6.0 / max(n_rows, 1))     # keep total height ≤ 6 in
        fig, ax = plt.subplots(figsize=(8, max(1.5, n_rows * row_h + 0.8)))
        im = ax.imshow(grid, cmap='RdBu_r', aspect='auto',
                       vmin=-abs(arr).max(), vmax=abs(arr).max())
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_xlabel('Feature mod 64', fontsize=8)
        ax.set_ylabel('Row', fontsize=8)
        ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    return _fig_to_src(fig)


def render_prediction(logits: torch.Tensor) -> str:
    probs   = torch.softmax(logits.float(), dim=0).numpy()
    log_np  = logits.float().numpy()
    colors  = ['#4c8bf5', '#ea4335', '#34a853']
    pred    = int(probs.argmax())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))
    ax1.bar(CLASS_NAMES, log_np, color=colors)
    ax1.set_title('Logits', fontsize=9)
    ax1.set_ylabel('Value', fontsize=8)
    ax1.tick_params(labelsize=8)

    bars = ax2.bar(CLASS_NAMES, probs, color=colors)
    ax2.set_ylim(0, 1.12)
    ax2.set_title('Softmax probabilities', fontsize=9)
    ax2.set_ylabel('Probability', fontsize=8)
    ax2.tick_params(labelsize=8)
    for bar, p in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{p:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle(f'Prediction: {CLASS_NAMES[pred]}  ({probs[pred]:.1%} confidence)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    return _fig_to_src(fig)

# ── Auto-detect model from params JSON ────────────────────────────────────────

def detect_model_from_weights(pth_filename: str) -> str | None:
    run_id = os.path.splitext(pth_filename)[0]
    candidate = os.path.join(PARAMS_DIR, f'params_{run_id}.json')
    if os.path.exists(candidate):
        arch = json.load(open(candidate)).get('architecture', '')
        for name in MODELS:
            if arch.startswith(name):
                return name
    return None

# ── App state ─────────────────────────────────────────────────────────────────

state: dict = {
    'model':      None,
    'layer_info': {},   # layer_name -> module type string
    'dataset':    None,
    'hooks':      [],
    'outputs':    {},   # layer_name -> tensor (batch dim squeezed)
}


def _clear_hooks():
    for h in state['hooks']:
        h.remove()
    state['hooks'].clear()
    state['outputs'].clear()


def _register_hooks(model: torch.nn.Module):
    _clear_hooks()
    layer_info = {}

    def make_hook(name):
        def hook(_, _inp, output):
            if isinstance(output, torch.Tensor):
                state['outputs'][name] = output.detach().cpu().squeeze(0)
        return hook

    for name, module in model.named_modules():
        if not list(module.children()):   # leaf modules only
            type_name = type(module).__name__
            layer_info[name] = type_name
            state['hooks'].append(module.register_forward_hook(make_hook(name)))

    state['layer_info'] = layer_info

# ── Event handlers (defined before UI so they can be passed as callbacks) ─────
# Note: references to UI elements (model_sel, auto_lbl, …) are resolved at
# call-time, not definition-time, so defining these before the UI block is fine.

def _on_weights_change(e):
    detected = detect_model_from_weights(e.value)
    if detected:
        model_sel.value = detected
        auto_lbl.set_text(f'Auto-detected: {detected}')
    else:
        auto_lbl.set_text('')


async def _load():
    _clear_hooks()
    state['model']   = None
    state['dataset'] = None
    sample_btn.disable()
    load_btn.props('loading')
    status_lbl.set_text('Loading weights…')
    vis_area.clear()
    await asyncio.sleep(0)   # yield so the UI can repaint before blocking calls

    loop = asyncio.get_event_loop()

    model_cls    = MODELS[model_sel.value]
    model        = model_cls(n_values=int(output_num.value)) if "MANO" in model_sel.value else model_cls()
    weights_path = os.path.join(MODELS_DIR, weights_sel.value)
    try:
        sd = await loop.run_in_executor(
            None, lambda: torch.load(weights_path, map_location='cpu', weights_only=True))
        model.load_state_dict(sd)
    except Exception as exc:
        status_lbl.set_text(f'Weight load error: {exc}')
        load_btn.props(remove='loading')
        return
    model.eval()
    _register_hooks(model)
    state['model'] = model

    status_lbl.set_text('Loading dataset…')
    await asyncio.sleep(0)
    try:
        ds = await loop.run_in_executor(
            None, lambda: torch.load(DATASETS[dataset_sel.value], weights_only=False))
        ds.augment = False
        state['dataset'] = ds
    except Exception as exc:
        status_lbl.set_text(f'Dataset load error: {exc}')
        load_btn.props(remove='loading')
        return

    load_btn.props(remove='loading')
    n = len(state['dataset'])
    status_lbl.set_text(f'{model_sel.value} | {weights_sel.value} | {n} samples ({dataset_sel.value})')
    ui.notify('Loaded — click "Random sample" to begin.', type='positive')
    sample_btn.enable()


async def _run_sample():
    model: torch.nn.Module = state['model']
    ds: GraspImageDataset  = state['dataset']
    if model is None or ds is None:
        return

    sample_btn.props('loading')
    await asyncio.sleep(0)

    idx = random.randint(0, len(ds) - 1)
    image, label = ds[idx]
    inp = image.float().unsqueeze(0)

    loop = asyncio.get_event_loop()

    def _compute():
        """Run inference + all matplotlib rendering in a worker thread."""
        state['outputs'].clear()
        with torch.no_grad():
            logits = model(inp).squeeze(0)

        renders = [('input', render_input(image))]

        layer_names = list(state['outputs'].keys())
        for layer_name in layer_names:
            tensor    = state['outputs'][layer_name]
            type_name = state['layer_info'].get(layer_name, '')
            is_last   = layer_name == layer_names[-1]

            if tensor.dim() == 3:
                C, H, W = tensor.shape
                title = f'{layer_name}  ({type_name})  —  {C} × {H}×{W}'
                img_src = render_feature_maps(tensor, title)
            elif tensor.dim() == 1:
                N = tensor.shape[0]
                title = f'{layer_name}  ({type_name})  —  {N} features'
                img_src = render_vector(tensor, title,
                                        class_labels=CLASS_NAMES if is_last else None)
            elif tensor.dim() == 2:
                H, W = tensor.shape
                title = f'{layer_name}  ({type_name})  —  {H}×{W}'
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(tensor.float().numpy(), cmap='viridis', aspect='auto')
                ax.set_title(title, fontsize=8)
                ax.axis('off')
                img_src = _fig_to_src(fig)
            else:
                continue

            renders.append(('layer', img_src))

        renders.append(('pred', render_prediction(logits)))
        return renders

    try:
        renders = await loop.run_in_executor(None, _compute)
    except Exception as exc:
        ui.notify(f'Error: {exc}', type='negative')
        sample_btn.props(remove='loading')
        return

    vis_area.clear()
    with vis_area:
        with ui.row().classes('items-center gap-4'):
            ui.label(f'Sample #{idx}').classes('text-lg font-bold')

        for kind, img_src in renders:
            if kind == 'input':
                with ui.card().classes('w-full p-2'):
                    ui.label('Input  (4 × H × W)').classes('text-xs font-semibold text-gray-500 mb-1')
                    ui.image(img_src).classes('w-full')
            elif kind == 'layer':
                with ui.card().classes('w-full p-2'):
                    ui.image(img_src).classes('w-full')
            elif kind == 'pred':
                with ui.card().classes('w-full p-2'):
                    ui.label('Prediction summary').classes('text-xs font-semibold text-gray-500 mb-1')
                    ui.image(img_src).classes('w-full')

    sample_btn.props(remove='loading')


# ── UI ────────────────────────────────────────────────────────────────────────

pth_files = sorted(os.listdir(MODELS_DIR))

with ui.header().classes('bg-indigo-900 text-white items-center px-4 py-2'):
    ui.label('GraspCNN — Layer Visualiser').classes('text-xl font-bold')

with ui.row().classes('w-full gap-4 p-4 items-start'):

    # ── Left panel ────────────────────────────────────────────────────────────
    with ui.card().classes('w-64 shrink-0 gap-1'):
        ui.label('Controls').classes('text-base font-semibold')

        weights_sel = ui.select(pth_files,
                                value=pth_files[0] if pth_files else None,
                                label='Weights file',
                                on_change=_on_weights_change).classes('w-full')
        auto_lbl    = ui.label('').classes('text-xs text-green-700 italic h-4')
        model_sel   = ui.select(list(MODELS), value='GraspCNNv1',
                                label='Model architecture').classes('w-full')
        output_num  = ui.number(label='No. output channels', min = 1, max = 15, value = 3)
        output_num.set_visibility(False)
        dataset_sel = ui.select(list(DATASETS), value='Test',
                                label='Dataset').classes('w-full')
        model_sel.on_value_change(lambda e: output_num.set_visibility("MANO" in e.value))
        ui.separator()
        load_btn   = ui.button('Load', icon='download', color='primary').classes('w-full')

        ui.separator()
        sample_btn = ui.button('Random sample', icon='shuffle', color='secondary').classes('w-full')
        sample_btn.disable()

        status_lbl = ui.label('Select a weights file and click Load.') \
                       .classes('text-xs text-gray-500 italic')

    # ── Right: visualisation area ─────────────────────────────────────────────
    with ui.column().classes('flex-1 min-w-0'):
        vis_area = ui.column().classes('w-full gap-3')


load_btn.on_click(_load)
sample_btn.on_click(_run_sample)

# Auto-detect on startup
_on_weights_change(type('_', (), {'value': weights_sel.value})())

ui.run(title='GraspCNN Layer Visualiser', port=8081, reload=False)
