"""
Batch GNN training scheduler.

Configure GLOBAL_CFG and SCHEDULE at the bottom of this file, then run:
    python train_gnn_batch.py

Seed strategy
─────────────
A master seed is used to pre-generate N trial-wise seeds once.
All models share the same seed list: trial t of model A and trial t of
model B use identical seeds, so results are reproducible regardless of
which model is trained first or in what order the schedule is defined.
"""

import datetime
import json
import os
import pickle
import random
import sys
import tempfile
import time
import traceback
from itertools import chain

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

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
from graspcnn.training import R2Accumulator

# ── Constants / paths ──────────────────────────────────────────────────────────

_DATA_DIR = os.path.expanduser('~/Documents/P10/GraspCNN/.data_cache')

DEFAULT_PATHS = dict(
    train_path=os.path.join(_DATA_DIR, 'data_src_train.pt'),
    val_path=os.path.join(_DATA_DIR,   'data_src_val.pt'),
    test_path=os.path.join(_DATA_DIR,  'data_src_test.pt'),
)

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


class WeightedLoss(nn.Module):
    """Pointwise loss weighted per output dimension by PCA explained-variance ratios."""

    def __init__(self, base_cls: type, weights: torch.Tensor):
        super().__init__()
        self.loss_fn = base_cls(reduction='none')
        # Normalize so weights sum to 1; shape [n_outputs]
        self.register_buffer('weights', weights / weights.sum())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # loss_fn returns [batch, n_outputs]; weight each column, then mean over all
        return (self.loss_fn(pred, target) * self.weights).mean()


# Allow TF32 on Ampere+ for ~3× matmul throughput with negligible accuracy loss.
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device found. This script requires a GPU.\n"
            f"  torch version  : {torch.__version__}\n"
            f"  CUDA available : {torch.cuda.is_available()}"
        )
    print(f"  CUDA device : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory // 2**20} MiB)")


# ── Reproducibility ────────────────────────────────────────────────────────────

def generate_trial_seeds(master_seed: int, n: int) -> list[int]:
    """Derive n reproducible trial seeds from a single master seed."""
    rng = random.Random(master_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


def set_seed(seed: int) -> torch.Generator:
    """Set all RNGs; return a torch.Generator seeded the same way."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Early stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop when val loss has not improved for `patience` consecutive epochs
    AND at least `min_epochs` have been completed.
    """

    def __init__(self, patience: int = 15, min_epochs: int = 20, delta: float = 1e-6):
        self.patience   = patience
        self.min_epochs = min_epochs
        self.delta      = delta
        self.best_loss  = float('inf')
        self.counter    = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Return True if training should be stopped."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience and epoch >= self.min_epochs


# ── Single trial ───────────────────────────────────────────────────────────────

def run_trial(
    *,
    model_name:    str,
    trial_idx:     int,
    seed:          int,
    cfg:           dict,
    train_dataset,
    val_dataset,
    test_dataset,
) -> dict:
    """Train one model trial; returns a summary dict."""

    g = set_seed(seed)

    dt     = datetime.datetime.now()
    run_id = f"{dt.strftime('%y%m%d_%H%M%S')}_{model_name}_t{trial_idx}"
    print(f"\n{'='*66}")
    print(f"  {run_id}  |  seed={seed}")
    print(f"{'='*66}")

    n_outputs = cfg['output_channels']
    n_in = cfg['n_in']

    for d in ('params', 'train_log', 'models', 'results'):
        os.makedirs(d, exist_ok=True)

    # ── Data loaders ───────────────────────────────────────────────────────────
    nw          = cfg.get('num_workers', 0)
    loader_kw   = dict(batch_size=cfg['batch_size'], pin_memory=True,
                       num_workers=nw, persistent_workers=(nw > 0))
    train_loader = GeoDataLoader(train_dataset, shuffle=True, generator=g, **loader_kw)
    val_loader   = GeoDataLoader(val_dataset,   shuffle=False,              **loader_kw)
    test_loader  = GeoDataLoader(test_dataset,  shuffle=False,              **loader_kw)

    # ── Model / optimiser ──────────────────────────────────────────────────────
    model = MODELS[model_name](
        dropoutgnn=cfg['dropout_gnn'],
        dropoutfc=cfg['dropout_fc'],
        n_in=n_in,
        n_out=n_outputs,
    ).to('cuda')
    torch.cuda.empty_cache()

    loss_weights = cfg.get('loss_weights')
    if loss_weights is not None and n_outputs > 1:
        criterion = WeightedLoss(LOSSES[cfg['loss']], loss_weights[:n_outputs]).to('cuda')
    else:
        criterion = LOSSES[cfg['loss']]()
    optimizer  = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    stopper    = EarlyStopping(patience=cfg['es_patience'], min_epochs=cfg['es_min_epochs'])
    use_amp    = cfg.get('use_amp', True)
    scaler     = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Persist run parameters ─────────────────────────────────────────────────
    json.dump(
        {
            'run_id':         run_id,
            'model':          model_name,
            'trial':          trial_idx,
            'seed':           seed,
            'date':           dt.isoformat(),
            'dropout_gnn':    cfg['dropout_gnn'],
            'dropout_fc':     cfg['dropout_fc'],
            'max_epochs':     cfg['epochs'],
            'loss_function':  cfg['loss'],
            'optimizer':      {'name': 'Adam', 'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']},
            'architecture':   str(model),
            'early_stopping': {'patience': cfg['es_patience'], 'min_epochs': cfg['es_min_epochs']},
        },
        open(f'params/params_{run_id}.json', 'w'),
        indent=2,
    )

    ckpt_path = f'models/{run_id}_best.pth'
    log_path  = f'train_log/train_log_{run_id}.txt'

    best_val_loss = float('inf')
    best_epoch    = -1
    r2_header     = '\t'.join(f'Val R2[{i}]' for i in range(n_outputs))
    trial_start   = time.time()

    # ── Training loop ──────────────────────────────────────────────────────────
    with open(log_path, 'w') as log:
        log.write(f'Epoch\tTrain Loss\tVal Loss\tVal R2 Mean\t{r2_header}\n')

        epoch_bar = tqdm(
            range(1, cfg['epochs'] + 1),
            desc=f'  epochs',
            unit='ep',
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            t0 = time.time()

            # train
            model.train()
            train_loss_acc = torch.tensor(0.0, device='cuda')
            for batch in train_loader:
                batch  = batch.to('cuda', non_blocking=True)
                batch.x = batch.x[:,:n_in]
                labels = batch.y[:, :n_outputs].float()
                optimizer.zero_grad()
                with torch.autocast('cuda', enabled=use_amp):
                    out  = model(batch).float()
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_acc += loss.detach()
            epoch_train_loss = (train_loss_acc / len(train_loader)).item()

            # validate
            model.eval()
            val_loss_sum = 0.0
            r2 = R2Accumulator(n_outputs, device='cuda')
            with torch.no_grad():
                for batch in val_loader:
                    batch  = batch.to('cuda', non_blocking=True)
                    batch.x = batch.x[:,:n_in]
                    labels = batch.y[:, :n_outputs].float()
                    with torch.autocast('cuda', enabled=use_amp):
                        out = model(batch).float()
                    val_loss_sum += criterion(out, labels).item()
                    r2.update(labels, out)

            r2_per_ch      = r2.r2_per_channel().tolist()
            r2_mean        = sum(r2_per_ch) / len(r2_per_ch)
            epoch_val_loss = val_loss_sum / len(val_loader)
            elapsed        = time.time() - t0

            r2_cols = '\t'.join(f'{v:.6f}' for v in r2_per_ch)
            log.write(f'{epoch}\t{epoch_train_loss:.6f}\t{epoch_val_loss:.6f}\t'
                      f'{r2_mean:.6f}\t{r2_cols}\n')
            log.flush()

            # save best checkpoint
            new_best = epoch_val_loss < best_val_loss
            if new_best:
                best_val_loss = epoch_val_loss
                best_epoch    = epoch
                torch.save(model.state_dict(), ckpt_path)

            epoch_bar.set_postfix(
                train=f'{epoch_train_loss:.5f}',
                val=f'{epoch_val_loss:.5f}' + (' *' if new_best else ''),
                R2=f'{r2_mean:.4f}',
                no_imp=stopper.counter,
                t=f'{elapsed:.1f}s',
            )

            if stopper.step(epoch_val_loss, epoch):
                tqdm.write(
                    f"  Early stop at epoch {epoch} — no improvement for "
                    f"{cfg['es_patience']} epochs (min_epochs={cfg['es_min_epochs']})."
                )
                break

        total_elapsed = time.time() - trial_start
        tqdm.write(
            f"  Trial done in {total_elapsed:.1f}s  |  "
            f"best val={best_val_loss:.5f} @ epoch {best_epoch}"
        )

    # ── Test evaluation on best checkpoint ─────────────────────────────────────
    print(f"  Loading best checkpoint (epoch {best_epoch}, "
          f"val_loss={best_val_loss:.5f}) for test evaluation...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    r2 = R2Accumulator(n_outputs, device='cuda')
    test_loss_sum = 0.0
    results  = []

    with torch.no_grad():
        for batch in test_loader:
            batch  = batch.to('cuda', non_blocking=True)
            batch.x = batch.x[:,:n_in]
            labels = batch.y[:, :n_outputs].float()
            with torch.autocast('cuda', enabled=use_amp):
                out = model(batch).float()
            results.extend(zip(labels.cpu().tolist(), out.cpu().tolist()))
            test_loss_sum += criterion(out, labels).item()
            r2.update(labels, out)

    r2_per_joint = r2.r2_per_channel()
    r2_mean_test = r2_per_joint.mean().item()
    test_loss    = test_loss_sum / len(test_loader)

    cols = [f'true_{i}' for i in range(n_outputs)] + [f'pred_{i}' for i in range(n_outputs)]
    rows = [list(chain(t, p)) for t, p in results]
    pd.DataFrame(rows, columns=cols).to_csv(f'results/results_{run_id}.csv', index=False)

    summary = dict(
        run_id=run_id,
        model=model_name,
        trial=trial_idx,
        seed=seed,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        r2_mean_test=r2_mean_test,
        r2_per_joint=r2_per_joint.tolist(),
    )
    json.dump(summary, open(f'results/summary_{run_id}.json', 'w'), indent=2)
    print(f"  Test: loss={test_loss:.5f}  R²={r2_mean_test:.4f}")
    return summary


# ── Batch runner ───────────────────────────────────────────────────────────────

def run_schedule(schedule: list[dict], global_cfg: dict) -> None:
    """
    Run all schedule entries sequentially.

    Each schedule entry supports:
        model_name  (str)   — required
        n_trials    (int)   — required
        loss        (str)   — optional, falls back to global_cfg['loss']
        dropout_gnn (float) — optional, falls back to global_cfg['dropout_gnn']
        dropout_fc  (float) — optional, falls back to global_cfg['dropout_fc']
    """

    _require_cuda()

    trial_seeds = generate_trial_seeds(global_cfg['master_seed'], global_cfg['n_seeds'])
    max_trials  = max(e['n_trials'] for e in schedule)
    if max_trials > global_cfg['n_seeds']:
        raise ValueError(
            f"n_trials={max_trials} exceeds n_seeds={global_cfg['n_seeds']}. "
            "Increase n_seeds in GLOBAL_CFG."
        )

    print("Loading datasets...")
    train_dataset = torch.load(global_cfg['train_path'], weights_only=False)
    val_dataset   = torch.load(global_cfg['val_path'],   weights_only=False)
    test_dataset  = torch.load(global_cfg['test_path'],  weights_only=False)
    print(f"Datasets loaded  "
          f"(train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)})")

    pca_path = global_cfg.get('pca_path')
    if pca_path:
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        evr = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32)
        print(f"PCA loaded: {len(evr)} components, "
              f"EVR={[round(v, 4) for v in evr.tolist()]}")
    else:
        evr = None

    all_summaries = []

    for entry in schedule:
        model_name  = entry['model_name']
        n_trials    = entry['n_trials']
        skip        = entry['skip']
        loss        = entry.get('loss',        global_cfg.get('loss',        'MSELoss'))
        dropout_gnn = entry.get('dropout_gnn', global_cfg.get('dropout_gnn', 0.1))
        dropout_fc  = entry.get('dropout_fc',  global_cfg.get('dropout_fc',  0.3))

        if model_name not in MODELS:
            print(f"WARNING: unknown model '{model_name}' — skipping.")
            continue
        if loss not in LOSSES:
            print(f"WARNING: unknown loss '{loss}' — skipping entry for {model_name}.")
            continue

        cfg = {
            **global_cfg,
            'loss':         loss,
            'dropout_gnn':  dropout_gnn,
            'dropout_fc':   dropout_fc,
            'loss_weights': evr,
        }

        for t in range(n_trials):
            if t<skip:continue
            summary = run_trial(
                model_name=model_name,
                trial_idx=t,
                seed=trial_seeds[t],
                cfg=cfg,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
            )
            all_summaries.append(summary)

    if all_summaries:
        ts  = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        out = f'results/batch_summary_{ts}.csv'
        pd.DataFrame(all_summaries).to_csv(out, index=False)
        print(f"\nBatch complete. Summary → {out}")
    else:
        print("\nBatch complete (no trials ran).")


# ══════════════════════════════════════════════════════════════════════════════
# Configure your experiment below, then run:  python train_gnn_batch.py
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_CFG = dict(
    # Reproducibility
    master_seed = 42,   # change for a different overall experiment
    n_seeds     = 10,   # pre-generate this many trial seeds; must be >= max n_trials

    # Architecture / training
    output_channels = 3,
    n_in            = 6,
    lr              = 1e-4,
    weight_decay    = 1e-4,
    batch_size      = 64,
    epochs          = 300,

    # Performance
    use_amp     = True,                              # AMP (FP16 forward pass, ~1.5–2× faster)
    num_workers = min(4, os.cpu_count() or 1),       # DataLoader worker processes

    # Early stopping
    es_patience  = 15,  # epochs without val-loss improvement before stopping
    es_min_epochs= 20,  # earliest epoch at which early stopping can trigger

    # Dataset paths (overrides DEFAULT_PATHS if specified)
    **DEFAULT_PATHS,

    # PCA for loss weighting (explained-variance ratio per output dimension)
    pca_path = os.path.expanduser('~/Documents/P10/hot3d/pca.pkl'),
)

SCHEDULE = [
    # Each entry trains `n_trials` independent runs of `model_name`.
    # Per-entry loss and dropout values override GLOBAL_CFG defaults.
    dict(model_name='MANOGraspGNNv1', n_trials=10, skip = 0, loss='MSELoss',  dropout_gnn=0.1, dropout_fc=0.3),
    dict(model_name='MANOGraspGNNv2', n_trials=10, skip = 0, loss='MSELoss',  dropout_gnn=0.1, dropout_fc=0.3),
    dict(model_name='MANOGraspGNNv4', n_trials=10, skip = 0, loss='MSELoss',  dropout_gnn=0.1, dropout_fc=0.3),
    dict(model_name='MANOGraspGNNv5', n_trials=10, skip = 0, loss='MSELoss',  dropout_gnn=0.1, dropout_fc=0.3),
    dict(model_name='MANOGraspGNNv7', n_trials=10, skip = 0, loss='MSELoss',  dropout_gnn=0.1, dropout_fc=0.3),
]


# ── Smoke test ────────────────────────────────────────────────────────────────

#: Models and loss functions exercised by the smoke test.
SMOKE_SCHEDULE = [
    dict(model_name='MANOGraspGNNv1', n_trials=2, loss='MSELoss',   dropout_gnn=0.1, dropout_fc=0.3),
    dict(model_name='MANOGraspGNNv2', n_trials=2, loss='HuberLoss', dropout_gnn=0.2, dropout_fc=0.4),
]


def smoke_test(data_paths: dict | None = None) -> bool:
    """
    Run a minimal end-to-end check (1 epoch, 2 trials, 2 models).

    All files are written to a temporary directory and deleted on exit.
    Returns True on success, False on failure.
    """
    pca_path = GLOBAL_CFG.get('pca_path')
    if pca_path:
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        evr = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32)
    else:
        evr = None

    cfg = dict(
        master_seed=42,
        n_seeds=10,
        output_channels=GLOBAL_CFG['output_channels'],
        n_in = GLOBAL_CFG['n_in'],
        lr=GLOBAL_CFG['lr'],
        weight_decay=GLOBAL_CFG['weight_decay'],
        batch_size=GLOBAL_CFG['batch_size'],
        epochs=1,
        es_patience=999,   # disable early stopping — not meaningful for 1 epoch
        es_min_epochs=999,
        loss_weights=evr,
        **(data_paths or DEFAULT_PATHS),
    )

    _require_cuda()

    original_dir = os.getcwd()
    passed = []
    failed = []

    with tempfile.TemporaryDirectory(prefix='grasp_smoke_') as tmp:
        os.chdir(tmp)
        try:
            print("\n" + "─" * 66)
            print("  SMOKE TEST — writing to temp dir, files discarded on exit")
            print("─" * 66)

            print("Loading datasets...")
            train_dataset = torch.load(cfg['train_path'], weights_only=False)
            val_dataset   = torch.load(cfg['val_path'],   weights_only=False)
            test_dataset  = torch.load(cfg['test_path'],  weights_only=False)
            print(f"  train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}")

            trial_seeds = generate_trial_seeds(cfg['master_seed'], cfg['n_seeds'])

            for entry in SMOKE_SCHEDULE:
                model_name  = entry['model_name']
                n_trials    = entry['n_trials']
                skip        = entry['skip']
                trial_cfg   = {**cfg,
                               'loss':        entry['loss'],
                               'dropout_gnn': entry['dropout_gnn'],
                               'dropout_fc':  entry['dropout_fc']}
                for t in range(n_trials):
                    if t < skip:continue
                    tag = f"{model_name} trial {t}"
                    try:
                        run_trial(
                            model_name=model_name,
                            trial_idx=t,
                            seed=trial_seeds[t],
                            cfg=trial_cfg,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            test_dataset=test_dataset,
                        )
                        passed.append(tag)
                    except Exception:
                        failed.append(tag)
                        print(f"\n  ERROR in {tag}:")
                        traceback.print_exc()

        finally:
            os.chdir(original_dir)

    print("\n" + "─" * 66)
    print(f"  SMOKE TEST RESULTS  ({len(passed)} passed, {len(failed)} failed)")
    for tag in passed:
        print(f"    PASS  {tag}")
    for tag in failed:
        print(f"    FAIL  {tag}")
    print("─" * 66)
    return len(failed) == 0


if __name__ == '__main__':
    if '--smoke-test' in sys.argv:
        ok = smoke_test()
        sys.exit(0 if ok else 1)
    else:
        run_schedule(SCHEDULE, GLOBAL_CFG)
