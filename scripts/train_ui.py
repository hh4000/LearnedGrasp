"""NiceGUI training interface for GraspCNN models."""

import datetime
import json
import os
import queue
import sys
import threading
import time
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from nicegui import ui

from graspcnn.losses import FocalLoss
from graspcnn.models import GraspCNNv1, GraspCNNv2, GraspCNNv3
from graspcnn.data import GraspImageDataset
from graspcnn.training import ConfigStore, DataLoaderGetter, chart_options


# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = {'GraspCNNv1': GraspCNNv1, 'GraspCNNv2': GraspCNNv2, "GraspCNNv3": GraspCNNv3}
LOSSES = {'CrossEntropy': nn.CrossEntropyLoss, 'FocalLoss': FocalLoss}

# Config lives at the repo root (one level above scripts/).
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_config.json')
_DATA_DIR   = os.path.expanduser('~/Documents/P10/hot3d/data')
DEFAULT_PATHS = {
    'train_path': os.path.join(_DATA_DIR, 'train_images.pt'),
    'val_path':   os.path.join(_DATA_DIR, 'val_images.pt'),
    'test_path':  os.path.join(_DATA_DIR, 'test_images.pt'),
}

_config = ConfigStore(CONFIG_FILE, DEFAULT_PATHS)


# ── Training thread ───────────────────────────────────────────────────────────

def training_thread(cfg: dict, result_q: queue.Queue, stop_evt: threading.Event):
    try:
        getter = DataLoaderGetter()
        sampler_val = cfg['sampler'] if cfg['sampler'] != 'None' else None
        train_loader = getter.get(
            path=cfg['train_path'],
            batch_size=cfg['batch_size'], half=True, device='cuda',
            sampler=sampler_val,
            shuffle=True if sampler_val is None else None,
            augment=cfg['augment'],
        )
        val_loader = getter.get(
            path=cfg['val_path'],
            batch_size=cfg['batch_size'], half=True, device='cpu', shuffle=False,
        )
        test_loader = getter.get(
            path=cfg['test_path'],
            batch_size=cfg['batch_size'], half=True, device='cpu', shuffle=False,
        )
        result_q.put(('status', 'Data loaded. Building model...'))

        model = MODELS[cfg['model']](dropout2d=cfg['dropout_2d'], dropoutfc=cfg['dropout_fc'])
        model.to('cuda')
        torch.backends.cudnn.benchmark = True

        loss_kwargs = {}
        if cfg['loss'] == 'FocalLoss':
            loss_kwargs['gamma'] = cfg['focal_gamma']
        if sampler_val is None:
            # No sampler → balance via loss weights (same formula as WeightedRandomSampler)
            ds = train_loader.dataset
            lbl = ds.labels if hasattr(ds, 'labels') else ds.tensors[1]
            classes = lbl.unique(sorted=True)
            class_counts = torch.stack([(lbl == c).sum() for c in classes]).float()
            print(f"Determined counts: {class_counts}")
            loss_kwargs['weight'] = (class_counts.min() / class_counts).to('cuda')
        criterion = LOSSES[cfg['loss']](**loss_kwargs)

        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg['scheduler_patience'])

        dt = datetime.datetime.now()
        run_id = dt.strftime('%y%m%d_%H%M%S')
        params = {
            'date': {'year': dt.year, 'month': dt.month, 'day': dt.day,
                     'hour': dt.hour, 'minute': dt.minute, 'second': dt.second},
            'dropout_fc': cfg['dropout_fc'], 'dropout_2d': cfg['dropout_2d'],
            'max_epochs': cfg['epochs'],
            'loss_function': {'name': cfg['loss'], 'kwargs': {
                k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in loss_kwargs.items()
            }},
            'optimizer': {'name': 'Adam', 'kwargs': {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']}},
            'architecture': str(model),
        }
        json.dump(params, open(f'params/params_{run_id}.json', 'w'), indent=2)
        result_q.put(('status', f'Run {run_id} — training started.'))
        ub_loader = DataLoader(train_loader.dataset, batch_size=cfg['batch_size'], shuffle=False)

        with open(f'train_log/train_log_{run_id}.txt', 'w') as log:
            log.write('Epoch\tTrain Loss (bias)\tTrain Loss (no bias)\tVal Loss\tVal Acc\tLR\n')

            for epoch in range(cfg['epochs']):
                if stop_evt.is_set():
                    result_q.put(('status', f'Stopped by user at epoch {epoch + 1}.'))
                    break

                t0 = time.time()
                model.train()
                train_loss = torch.tensor(0.0, device='cuda')
                for inputs, labels in train_loader:
                    inputs = inputs.to('cuda', non_blocking=True).float()
                    labels = labels.to('cuda', non_blocking=True)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach()
                epoch_train_loss = (train_loss / len(train_loader)).item()
                t1 = time.time()

                # unbiased eval on training set (no augmentation, no sampler)
                model.eval()
                if hasattr(train_loader.dataset, 'augment'):
                    train_loader.dataset.augment = False
                train_loss_ub = torch.tensor(0.0, device='cuda')
                with torch.no_grad():
                    for inputs, labels in ub_loader:
                        inputs = inputs.to('cuda', non_blocking=True).float()
                        labels = labels.to('cuda', non_blocking=True)
                        train_loss_ub += criterion(model(inputs), labels).detach()
                epoch_train_loss_ub = (train_loss_ub / len(ub_loader)).item()
                if cfg['augment'] and hasattr(train_loader.dataset, 'augment'):
                    train_loader.dataset.augment = True

                val_loss = torch.tensor(0.0, device='cuda')
                correct = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to('cuda', non_blocking=True).float()
                        labels = labels.to('cuda', non_blocking=True)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).detach()
                        correct += (outputs.argmax(1) == labels).sum().detach()
                epoch_val_loss = (val_loss / len(val_loader)).item()
                epoch_val_acc = (correct / len(val_loader.dataset)).item()
                t2 = time.time()
                current_lr = optimizer.param_groups[0]['lr']

                result_q.put(('epoch', {
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'train_loss_ub': epoch_train_loss_ub,
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                    'lr': current_lr,
                    'time': t2 - t0,
                }))
                log.write(f'{epoch + 1}\t{epoch_train_loss}\t{epoch_train_loss_ub}\t'
                          f'{epoch_val_loss}\t{epoch_val_acc}\t{current_lr}\n')
                log.flush()

                scheduler.step(epoch_val_loss)
                if epoch_val_loss >= cfg['overfit_mult'] * epoch_train_loss_ub and epoch + 1 > 25:
                    result_q.put(('status',
                        f'Early stop: val={epoch_val_loss:.4f} >= '
                        f'{cfg["overfit_mult"]} × train={epoch_train_loss_ub:.4f}'))
                    break

        torch.save(model.state_dict(), f'models/{run_id}.pth')
        model.eval()
        test_loss_sum = 0.0
        cm = torch.zeros((3, 3), dtype=torch.int64)
        results = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs: torch.Tensor = inputs.to('cuda', non_blocking=True).float()
                labels: torch.Tensor = labels.to('cuda', non_blocking=True)
                outputs = model(inputs)
                test_loss_sum += criterion(outputs, labels).item()
                preds = outputs.argmax(1).cpu()
                results.append(pd.DataFrame(dict(label=labels, pred=preds)))
                cm += torch.zeros(3, 3, dtype=torch.int64).index_put_(
                    (labels.cpu(), preds), torch.ones(len(preds), dtype=torch.int64), accumulate=True)
        pd.concat(results, ignore_index = True).to_csv(f'results/results_{run_id}.csv', index=False)
        result_q.put(('done', {
            'cm': cm.tolist(),
            'test_loss': test_loss_sum / len(test_loader),
            'run_id': run_id,
        }))

    except Exception as e:
        result_q.put(('error', str(e)))
        raise


# ── Global queue / event (shared across the single-page app) ──────────────────

result_q: queue.Queue = queue.Queue()
stop_evt: threading.Event = threading.Event()
train_thread: threading.Thread | None = None

# ── UI ────────────────────────────────────────────────────────────────────────

_cfg = _config.load()
epochs_x: list[int] = []

with ui.header().classes('bg-blue-900 text-white items-center px-4 py-2'):
    ui.label('GraspCNN Training').classes('text-xl font-bold')
    reload_btn = ui.button(icon="restart_alt")

with ui.tabs().classes('w-full bg-blue-50') as tabs:
    tab_train   = ui.tab('Training',  icon='fitness_center')
    tab_dataset = ui.tab('Dataset',   icon='folder_open')

with ui.tab_panels(tabs, value=tab_train).classes('w-full'):

    # ══ Training tab ══════════════════════════════════════════════════════════
    with ui.tab_panel(tab_train):
        with ui.row().classes('w-full gap-4 p-4 items-start'):

            # ── Left: config card ─────────────────────────────────────────────
            with ui.card().classes('w-72 shrink-0 gap-1'):
                ui.label('Configuration').classes('text-base font-semibold')

                model_sel   = ui.select(list(MODELS), value='GraspCNNv1', label='Model').classes('w-full')
                gamma_row   = ui.number('Focal γ', value=2.0, step=0.5, min=0.0).classes('w-full')
                gamma_row.set_visibility(False)
                loss_sel    = ui.select(list(LOSSES), value='CrossEntropy', label='Loss Function',
                                        on_change=lambda e: gamma_row.set_visibility(e.value == 'FocalLoss'),
                                        ).classes('w-full')

                ui.separator()
                dropout_fc  = ui.number('Dropout FC',    value=0.5,  step=0.05, min=0.0, max=1.0).classes('w-full')
                dropout_2d  = ui.number('Dropout 2D',    value=0.3,  step=0.05, min=0.0, max=1.0).classes('w-full')
                lr_inp      = ui.number('Learning Rate', value=1e-3, step=1e-4, min=1e-7, format='%.7f').classes('w-full')
                wd_inp      = ui.number('Weight Decay',  value=1e-4, step=1e-5, min=0.0,  format='%.7f').classes('w-full')
                epochs_inp  = ui.number('Max Epochs',    value=1000, step=10,   min=1).classes('w-full')
                batch_inp   = ui.number('Batch Size',    value=32,   step=8,    min=1).classes('w-full')
                overfit_inp   = ui.number('Overfit Mult.',      value=2.0,  step=0.1, min=1.0).classes('w-full')
                scheduler_inp = ui.number('Scheduler Patience', value=10,   step=1,   min=1).classes('w-full')

                ui.separator()
                augment_chk = ui.checkbox('Augment train data', value=True)
                sampler_sel = ui.select(
                    ['None', 'WeightedRandomSampler'],
                    value='WeightedRandomSampler',
                    label='Sampler',
                ).classes('w-full')

                ui.separator()
                status_lbl = ui.label('Ready').classes('text-sm text-gray-500 italic')

            # ── Right: controls + charts ──────────────────────────────────────
            with ui.column().classes('flex-1 min-w-0 gap-4'):

                with ui.row().classes('gap-2'):
                    start_btn = ui.button('Start Training', icon='play_arrow', color='positive')
                    stop_btn  = ui.button('Stop',           icon='stop',        color='negative')
                    stop_btn.disable()

                loss_chart = ui.echart(chart_options(
                    'Loss',
                    ['Train (bias)', 'Train (no bias)', 'Val Loss'],
                    'Loss',
                )).classes('w-full h-64')

                with ui.row().classes('w-full gap-4'):
                    acc_chart = ui.echart(chart_options(
                        'Validation Accuracy (balanced)', ['Val Acc'], 'Accuracy',
                    )).classes('flex-1 h-48')
                    lr_chart = ui.echart(chart_options(
                        'Learning Rate', ['LR'], 'LR',
                    )).classes('flex-1 h-48')

    # ══ Dataset tab ═══════════════════════════════════════════════════════════
    with ui.tab_panel(tab_dataset):
        with ui.card().classes('max-w-xl m-4 gap-3'):
            ui.label('Dataset file paths').classes('text-base font-semibold')
            ui.label('Paths are saved automatically and reloaded on next launch.') \
              .classes('text-sm text-gray-500')
            ui.separator()
            path_train = ui.input('Train set (.pt)',      value=_cfg['train_path']).classes('w-full')
            path_val   = ui.input('Validation set (.pt)', value=_cfg['val_path']).classes('w-full')
            path_test  = ui.input('Test set (.pt)',        value=_cfg['test_path']).classes('w-full')

            def _check_paths():
                ok = True
                for inp, key in ((path_train, 'train'), (path_val, 'val'), (path_test, 'test')):
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
                ui.button('Verify', icon='check_circle', color='secondary',
                          on_click=_check_paths)
                ui.button('Save',   icon='save',         color='primary',
                          on_click=_save_paths)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_cfg() -> dict:
    return dict(
        model=model_sel.value,
        loss=loss_sel.value,
        focal_gamma=float(gamma_row.value),
        dropout_fc=float(dropout_fc.value),
        dropout_2d=float(dropout_2d.value),
        lr=float(lr_inp.value),
        weight_decay=float(wd_inp.value),
        epochs=int(epochs_inp.value),
        batch_size=int(batch_inp.value),
        overfit_mult=float(overfit_inp.value),
        scheduler_patience=int(scheduler_inp.value),
        augment=augment_chk.value,
        sampler=sampler_sel.value,
        train_path=os.path.expanduser(path_train.value),
        val_path=os.path.expanduser(path_val.value),
        test_path=os.path.expanduser(path_test.value),
    )


def _reset_charts():
    epochs_x.clear()
    for chart in (loss_chart, acc_chart, lr_chart):
        chart.options['xAxis']['data'] = []
        for s in chart.options['series']:
            s['data'] = []
        chart.update()


def _push_epoch(data: dict):
    e = data['epoch']
    epochs_x.append(e)

    loss_chart.options['xAxis']['data'] = epochs_x.copy()
    loss_chart.options['series'][0]['data'].append(round(data['train_loss'],    6))
    loss_chart.options['series'][1]['data'].append(round(data['train_loss_ub'], 6))
    loss_chart.options['series'][2]['data'].append(round(data['val_loss'],      6))
    loss_chart.update()

    acc_chart.options['xAxis']['data'] = epochs_x.copy()
    acc_chart.options['series'][0]['data'].append(round(data['val_acc'], 6))
    acc_chart.update()

    lr_chart.options['xAxis']['data'] = epochs_x.copy()
    lr_chart.options['series'][0]['data'].append(data['lr'])
    lr_chart.update()

    status_lbl.set_text(
        f"Epoch {e} | train={data['train_loss']:.4f} | "
        f"val={data['val_loss']:.4f} | acc={data['val_acc']:.4f} | "
        f"{data['time']:.1f}s/epoch"
    )


def _show_done_dialog(data: dict):
    cm       = data['cm']
    run_id   = data['run_id']
    test_loss = data['test_loss']
    class_names = ['Lateral', 'Pinch', 'Power']

    with ui.dialog() as dlg, ui.card().classes('p-6 gap-3 min-w-[420px]'):
        ui.label('Training Complete!').classes('text-2xl font-bold text-green-600')
        ui.label(f'Run ID: {run_id}').classes('text-sm text-gray-500')
        ui.label(f'Test Loss: {test_loss:.4f}').classes('text-lg font-semibold')

        ui.separator()
        ui.label('Confusion Matrix').classes('font-semibold')

        with ui.grid(columns=4).classes('gap-1 text-sm'):
            ui.label('')  # top-left corner
            for name in class_names:
                ui.label(f'Pred: {name}').classes('text-center font-medium text-blue-700')
            for i, row in enumerate(cm):
                ui.label(f'True: {class_names[i]}').classes('font-medium text-blue-700')
                for j, val in enumerate(row):
                    bg = 'bg-green-200' if i == j else 'bg-red-50'
                    ui.label(str(val)).classes(f'text-center p-1 rounded {bg}')

        ui.separator()
        ui.button('Close', on_click=dlg.close).classes('w-full')

    dlg.open()


def _on_done(data: dict):
    start_btn.enable()
    stop_btn.disable()
    status_lbl.set_text(f"Done — test loss: {data['test_loss']:.4f}  |  model saved as {data['run_id']}.pth")
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
        target=training_thread, args=(cfg, result_q, stop_evt), daemon=True)
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

ui.run(title='GraspCNN Training', port=8080, reload=False)
