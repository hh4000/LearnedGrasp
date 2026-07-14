"""NiceGUI training interface for MANOGraspGNN models."""

import os
import queue
import sys
import threading

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from nicegui import ui

import graspcnn.data.mesh_set as grasp_mesh_set
from graspcnn.models import (
    MANOGraspGNNv1,
    MANOGraspGNNv2,
    MANOGraspGNNv3,
    MANOGraspGNNv4,
    MANOGraspGNNv5,
    MANOGraspGNNv6,
    MANOGraspGNNv7,
)
from graspcnn.training import ConfigStore, Trainer, chart_options

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = {
    'MANOGraspGNNv1': MANOGraspGNNv1,
    'MANOGraspGNNv2': MANOGraspGNNv2,
    'MANOGraspGNNv3': MANOGraspGNNv3,
    'MANOGraspGNNv4': MANOGraspGNNv4,
    'MANOGraspGNNv5': MANOGraspGNNv5,
    'MANOGraspGNNv6': MANOGraspGNNv6,
    'MANOGraspGNNv7': MANOGraspGNNv7,
}
LOSSES = {
    'MSELoss':   nn.MSELoss,
    'HuberLoss': nn.HuberLoss,
}

# Config lives at the repo root (one level above scripts/).
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_gnn_config.json')
_DATA_DIR   = os.path.expanduser('~/Documents/P10/hot3d/data')
DEFAULT_PATHS = {
    'train_path': os.path.join(_DATA_DIR, 'train_gnn.pt'),
    'val_path':   os.path.join(_DATA_DIR, 'val_gnn.pt'),
    'test_path':  os.path.join(_DATA_DIR, 'test_gnn.pt'),
}

_config = ConfigStore(CONFIG_FILE, DEFAULT_PATHS)


# ── Training thread ───────────────────────────────────────────────────────────

class _GNNTrainer(Trainer):
    """Regression trainer for graph batches (torch_geometric ``Data``)."""

    def forward_batch(self, batch):
        batch = batch.to(self.device)
        labels = batch.y[:, :self.n_outputs].float()
        return self.model(batch).float(), labels


def training_thread(cfg: dict, result_q: queue.Queue, stop_evt: threading.Event):
    try:
        train_dataset:  grasp_mesh_set.PrecachedMANOGraspDataset = torch.load(cfg['train_path'], weights_only=False)
        val_dataset:    grasp_mesh_set.PrecachedMANOGraspDataset = torch.load(cfg['val_path'],   weights_only=False)
        test_dataset:   grasp_mesh_set.PrecachedMANOGraspDataset = torch.load(cfg['test_path'],  weights_only=False)
        result_q.put(('status', 'Data loaded. Building model...'))

        train_loader = GeoDataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
        val_loader   = GeoDataLoader(val_dataset,   batch_size=cfg['batch_size'], shuffle=False)
        test_loader  = GeoDataLoader(test_dataset,  batch_size=cfg['batch_size'], shuffle=False)

        model = MODELS[cfg['model']](
            dropoutgnn=cfg['dropout_gnn'],
            dropoutfc=cfg['dropout_fc'],
            n_in = train_dataset.data_dims,
            n_out=cfg['output_channels'],
        )
        model.to('cuda')
        torch.cuda.empty_cache()

        criterion = LOSSES[cfg['loss']]()
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=cfg['scheduler_patience']
        )

        trainer = _GNNTrainer(model, criterion, optimizer, scheduler,
                              result_q=result_q, stop_evt=stop_evt,
                              n_outputs=cfg['output_channels'], device='cuda')
        trainer.dump_params(trainer.base_params(cfg, extra={
            'dropout_gnn': cfg['dropout_gnn'],
            'loss_function': {'name': cfg['loss']},
        }))
        result_q.put(('status', f'Run {trainer.run_id} — training started.'))

        with open(trainer.log_path, 'w') as log:
            trainer.log_header(log)
            trainer.fit(train_loader, val_loader, cfg['epochs'], log)

        trainer.save_checkpoint()
        trainer.test(test_loader)

    except Exception as e:
        result_q.put(('error', str(e)))
        raise


# ── Global queue / event ──────────────────────────────────────────────────────

result_q: queue.Queue = queue.Queue()
stop_evt:  threading.Event = threading.Event()
train_thread: threading.Thread | None = None

# ── UI ────────────────────────────────────────────────────────────────────────

_cfg = _config.load()
epochs_x: list[int] = []

with ui.header().classes('bg-indigo-900 text-white items-center px-4 py-2'):
    ui.label('GraspGNN Training').classes('text-xl font-bold')
    reload_btn = ui.button(icon='restart_alt')

with ui.tabs().classes('w-full bg-indigo-50') as tabs:
    tab_train   = ui.tab('Training', icon='fitness_center')
    tab_dataset = ui.tab('Dataset',  icon='folder_open')

with ui.tab_panels(tabs, value=tab_train).classes('w-full'):

    # ══ Training tab ══════════════════════════════════════════════════════════
    with ui.tab_panel(tab_train):
        with ui.row().classes('w-full gap-4 p-4 items-start'):

            # ── Left: config card ─────────────────────────────────────────────
            with ui.card().classes('w-72 shrink-0 gap-1'):
                ui.label('Configuration').classes('text-base font-semibold')

                model_sel     = ui.select(list(MODELS), value='MANOGraspGNNv1', label='Model').classes('w-full')
                out_nums      = ui.number('Output values', min=1, max=15, value=3, step=1).classes('w-full')
                loss_sel      = ui.select(list(LOSSES), value='MSELoss', label='Loss Function').classes('w-full')

                ui.separator()
                dropout_gnn   = ui.number('Dropout GNN', value=0.1, step=0.05, min=0.0, max=0.9).classes('w-full')
                dropout_fc    = ui.number('Dropout FC',  value=0.3, step=0.05, min=0.0, max=0.9).classes('w-full')
                lr_inp        = ui.number('Learning Rate', value=3e-4, step=1e-4, min=1e-7, format='%.7f').classes('w-full')
                wd_inp        = ui.number('Weight Decay',  value=1e-4, step=1e-5, min=0.0,  format='%.7f').classes('w-full')
                epochs_inp    = ui.number('Max Epochs',    value=300,  step=10,   min=1).classes('w-full')
                batch_inp     = ui.number('Batch Size',    value=16,   step=4,    min=1).classes('w-full')
                scheduler_inp = ui.number('Scheduler Patience', value=20, step=1, min=1).classes('w-full')

                ui.separator()
                status_lbl = ui.label('Ready').classes('text-sm text-gray-500 italic')

            # ── Right: controls + charts ──────────────────────────────────────
            with ui.column().classes('flex-1 min-w-0 gap-4'):

                with ui.row().classes('gap-2'):
                    start_btn = ui.button('Start Training', icon='play_arrow', color='positive')
                    stop_btn  = ui.button('Stop',           icon='stop',        color='negative')
                    stop_btn.disable()

                loss_chart = ui.echart(chart_options(
                    'Loss', ['Train Loss', 'Val Loss'], 'Loss',
                )).classes('w-full h-64')

                with ui.row().classes('w-full gap-4'):
                    acc_chart = ui.echart(chart_options(
                        'Val R² per dim', [f'dim {i}' for i in range(int(out_nums.value))], 'R²',
                    )).classes('flex-1 h-48')
                    lr_chart = ui.echart(chart_options(
                        'Learning Rate', ['LR'], 'LR',
                    )).classes('flex-1 h-48')

    # ══ Dataset tab ═══════════════════════════════════════════════════════════
    with ui.tab_panel(tab_dataset):
        with ui.card().classes('max-w-xl m-4 gap-3'):
            ui.label('Dataset file paths').classes('text-base font-semibold')
            ui.label('Each .pt file should be a serialised PrecachedMANOGraspDataset.') \
              .classes('text-sm text-gray-500')
            ui.separator()
            path_train = ui.input('Train set (.pt)',      value=_cfg['train_path']).classes('w-full')
            path_val   = ui.input('Validation set (.pt)', value=_cfg['val_path']).classes('w-full')
            path_test  = ui.input('Test set (.pt)',        value=_cfg['test_path']).classes('w-full')

            def _check_paths():
                ok = True
                for inp in (path_train, path_val, path_test):
                    exists = os.path.isfile(os.path.expanduser(inp.value))
                    inp.props(f'{"" if exists else "error"}')
                    if not exists:
                        ok = False
                return ok

            def _save_paths():
                paths = {
                    'train_path': path_train.value,
                    'val_path':   path_val.value,
                    'test_path':  path_test.value,
                }
                if _check_paths():
                    _config.save(paths)
                    ui.notify('Paths saved.', type='positive')
                else:
                    ui.notify('One or more paths not found — not saved.', type='warning')

            with ui.row().classes('gap-2 mt-2'):
                ui.button('Verify', icon='check_circle', color='secondary', on_click=_check_paths)
                ui.button('Save',   icon='save',         color='primary',   on_click=_save_paths)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_cfg() -> dict:
    return dict(
        model=model_sel.value,
        output_channels=int(out_nums.value),
        loss=loss_sel.value,
        dropout_gnn=float(dropout_gnn.value),
        dropout_fc=float(dropout_fc.value),
        lr=float(lr_inp.value),
        weight_decay=float(wd_inp.value),
        epochs=int(epochs_inp.value),
        batch_size=int(batch_inp.value),
        scheduler_patience=int(scheduler_inp.value),
        train_path=os.path.expanduser(path_train.value),
        val_path=os.path.expanduser(path_val.value),
        test_path=os.path.expanduser(path_test.value),
    )


def _reset_charts():
    epochs_x.clear()
    n = int(out_nums.value)
    for chart in (loss_chart, lr_chart):
        chart.options['xAxis']['data'] = []
        for s in chart.options['series']:
            s['data'] = []
        chart.update()
    acc_chart.options.update(chart_options(
        'Val R² per dim', [f'dim {i}' for i in range(n)], 'R²',
    ))
    acc_chart.update()


def _push_epoch(data: dict):
    e = data['epoch']
    epochs_x.append(e)

    loss_chart.options['xAxis']['data'] = epochs_x.copy()
    loss_chart.options['series'][0]['data'].append(round(data['train_loss'], 6))
    loss_chart.options['series'][1]['data'].append(round(data['val_loss'],   6))
    loss_chart.update()

    acc_chart.options['xAxis']['data'] = epochs_x.copy()
    for i, r2 in enumerate(data['r2_per_ch']):
        acc_chart.options['series'][i]['data'].append(round(r2, 6))
    acc_chart.update()

    lr_chart.options['xAxis']['data'] = epochs_x.copy()
    lr_chart.options['series'][0]['data'].append(data['lr'])
    lr_chart.update()

    status_lbl.set_text(
        f"Epoch {e} | train={data['train_loss']:.4f} | "
        f"val={data['val_loss']:.4f} | R²={data['r2_mean']:.4f} | "
        f"{data['time']:.1f}s/epoch"
    )


def _show_done_dialog(data: dict):
    r2_per_joint = data['r2_per_joint']
    r2_mean      = data['r2_mean']
    run_id       = data['run_id']
    test_loss    = data['test_loss']

    with ui.dialog() as dlg, ui.card().classes('p-6 gap-3 min-w-[420px]'):
        ui.label('Training Complete!').classes('text-2xl font-bold text-green-600')
        ui.label(f'Run ID: {run_id}').classes('text-sm text-gray-500')
        ui.label(f'Test Loss: {test_loss:.4f}').classes('text-lg font-semibold')
        ui.label(f'Mean R²: {r2_mean:.4f}').classes('text-lg font-semibold')

        ui.separator()
        ui.label('R² per Output').classes('font-semibold')

        with ui.grid(columns=2).classes('gap-1 text-sm w-full'):
            ui.label('Output').classes('font-medium text-indigo-700')
            ui.label('R²').classes('font-medium text-indigo-700 text-center')
            for i, r2 in enumerate(r2_per_joint):
                ui.label(f'Output {i+1}').classes('text-gray-700')
                bg = 'bg-green-200' if r2 >= 0.7 else 'bg-yellow-100' if r2 >= 0.4 else 'bg-red-100'
                ui.label(f'{r2:.4f}').classes(f'text-center p-1 rounded {bg}')

        ui.separator()
        ui.button('Close', on_click=dlg.close).classes('w-full')

    dlg.open()


def _on_done(data: dict):
    start_btn.enable()
    stop_btn.disable()
    status_lbl.set_text(
        f"Done — test loss: {data['test_loss']:.4f}  |  model saved as {data['run_id']}.pth"
    )
    _show_done_dialog(data)


# ── Queue polling timer ───────────────────────────────────────────────────────

def _poll_queue():
    while not result_q.empty():
        try:
            kind, payload = result_q.get_nowait()
        except queue.Empty:
            break
        if kind == 'epoch':
            _push_epoch(payload)
        elif kind == 'status':
            status_lbl.set_text(payload)
        elif kind == 'done':
            _on_done(payload)
        elif kind == 'error':
            ui.notify(f'Training error: {payload}', type='negative', timeout=0)
            status_lbl.set_text(f'Error: {payload}')
            start_btn.enable()
            stop_btn.disable()


ui.timer(0.5, _poll_queue)


# ── Button handlers ───────────────────────────────────────────────────────────

def _start_training():
    global train_thread
    if train_thread and train_thread.is_alive():
        return
    _reset_charts()
    stop_evt.clear()
    cfg = _collect_cfg()
    start_btn.disable()
    stop_btn.enable()
    status_lbl.set_text('Loading data...')
    train_thread = threading.Thread(
        target=training_thread, args=(cfg, result_q, stop_evt), daemon=True
    )
    train_thread.start()


def _stop_training():
    stop_evt.set()
    stop_btn.disable()
    status_lbl.set_text('Stopping after current epoch...')


def _restart_server():
    os.execv(sys.executable, [sys.executable] + sys.argv)


reload_btn.on_click(_restart_server)
ui.timer(0.5, lambda: reload_btn.set_enabled(train_thread is None or not train_thread.is_alive()))
start_btn.on_click(_start_training)
stop_btn.on_click(_stop_training)


# ── Run ───────────────────────────────────────────────────────────────────────

ui.run(title='GraspGNN Training', port=8081, reload=False)
