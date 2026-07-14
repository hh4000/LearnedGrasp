"""Shared training infrastructure for the GraspCNN entrypoint scripts.

The entrypoints under ``scripts/`` (``train_ui``, ``train_regression_ui``,
``train_gnn_ui``, ``train_gnn_batch``) historically duplicated a large amount of
run bookkeeping and — for the two regression trainers — an almost identical
epoch loop. This module factors the shared, behaviour-preserving pieces:

- :class:`DataLoaderGetter` — the generic image-dataset loader builder.
- :class:`R2Accumulator`   — streaming coefficient-of-determination over channels.
- :class:`Trainer`         — run bookkeeping (ids, params/log/checkpoint paths,
  result-queue protocol, early-stop/scheduler plumbing) plus a reusable
  regression fit/test loop driven by small per-script hooks.

Each script keeps its own task-specific model/loss/forward logic; the numeric
behaviour of the original loops is preserved.
"""

from __future__ import annotations

import datetime
import json
import queue
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


# ── Data loading ────────────────────────────────────────────────────────────────

class DataLoaderGetter:
    """Build a ``DataLoader`` from a pickled dataset, with optional class balancing.

    Handles the various dataset shapes used across the project (objects exposing
    ``labels``/``images``/``half``/``to``/``augment`` as well as plain
    ``TensorDataset``\\ s).
    """

    SAMPLERS = {'WeightedRandomSampler': '_get_weighted_random_sampler'}

    def get(self, path, *, batch_size=32, half=False, device='cuda',
            shuffle=None, sampler=None, augment=False, **_):
        dataset = torch.load(path, weights_only=False)
        loader_kwargs = {'batch_size': batch_size}
        if half and hasattr(dataset, 'half'):
            dataset.half()
        elif half and hasattr(dataset, 'images'):
            dataset.images = dataset.images.half()
        elif half and hasattr(dataset, 'tensors'):
            dataset.tensors = (dataset.tensors[0].half(), dataset.tensors[1])
        if hasattr(dataset, 'to'):
            dataset.to(device)
        elif hasattr(dataset, 'tensors'):
            dataset.tensors = tuple(t.to(device) for t in dataset.tensors)
        if hasattr(dataset, 'augment'):
            dataset.augment = augment
        if shuffle is not None and sampler is not None:
            raise ValueError("'shuffle' and 'sampler' are mutually exclusive.")
        elif sampler is not None:
            labels = dataset.labels if hasattr(dataset, 'labels') else dataset.tensors[1]
            loader_kwargs['sampler'] = getattr(self, self.SAMPLERS[sampler])(labels)
        elif shuffle is not None:
            loader_kwargs['shuffle'] = shuffle
        else:
            loader_kwargs['shuffle'] = False
        return DataLoader(dataset, **loader_kwargs)

    @staticmethod
    def _get_weighted_random_sampler(labels):
        classes = labels.unique(sorted=True)
        class_counts = torch.Tensor([(labels == cls).sum() for cls in classes]).to(labels.device)
        w = class_counts.min() / class_counts
        sample_weights = w[labels]
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# ── Metrics ─────────────────────────────────────────────────────────────────────

class R2Accumulator:
    """Streaming per-channel R² (coefficient of determination).

    Accumulates residual and total sums of squares over batches using the
    computational formula, so the whole dataset never has to be held in memory.
    Reproduces the inline computation the regression trainers used previously.
    """

    def __init__(self, n_outputs: int, device='cuda'):
        self.n_outputs = n_outputs
        self.ss_res = torch.zeros(n_outputs, device=device)
        self.sum_t = torch.zeros(n_outputs, device=device)
        self.sum_sq_t = torch.zeros(n_outputs, device=device)
        self.n_total = 0

    def update(self, target: torch.Tensor, pred: torch.Tensor) -> None:
        self.ss_res += ((target - pred) ** 2).sum(dim=0)
        self.sum_t += target.sum(dim=0)
        self.sum_sq_t += (target ** 2).sum(dim=0)
        self.n_total += target.shape[0]

    def r2_per_channel(self) -> torch.Tensor:
        target_mean = self.sum_t / self.n_total
        ss_tot = (self.sum_sq_t - self.n_total * target_mean ** 2).clamp(min=0)
        return 1 - self.ss_res / ss_tot.clamp(min=1e-6)


# ── Trainer ─────────────────────────────────────────────────────────────────────

class Trainer:
    """Run bookkeeping + a reusable regression training loop.

    Subclasses/callers supply the task-specific bits via hooks:

    - :meth:`forward_batch` — turn a loader item into ``(prediction, target)``.
    - :meth:`reduce_loss`   — post-process the raw criterion output to a scalar.
    - :meth:`build_params`  — the JSON-serialisable parameter record to persist.

    Bookkeeping helpers (:meth:`dump_params`, :meth:`save_checkpoint`,
    :meth:`emit`, :meth:`stop_requested`) are usable on their own by trainers
    whose loop differs (e.g. the classification trainer).
    """

    def __init__(self, model, criterion, optimizer, scheduler, *,
                 result_q: queue.Queue, stop_evt: threading.Event,
                 n_outputs: int, device: str = 'cuda', root: str | Path = '.'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.result_q = result_q
        self.stop_evt = stop_evt
        self.n_outputs = n_outputs
        self.device = device
        self.root = Path(root)
        self.started_at = datetime.datetime.now()
        self.run_id = self.started_at.strftime('%y%m%d_%H%M%S')

    # -- paths -------------------------------------------------------------------
    def _path(self, subdir: str, filename: str) -> Path:
        d = self.root / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d / filename

    @property
    def params_path(self) -> Path:
        return self._path('params', f'params_{self.run_id}.json')

    @property
    def log_path(self) -> Path:
        return self._path('train_log', f'train_log_{self.run_id}.txt')

    @property
    def results_path(self) -> Path:
        return self._path('results', f'results_{self.run_id}.csv')

    @property
    def checkpoint_path(self) -> Path:
        return self._path('models', f'{self.run_id}.pth')

    # -- bookkeeping helpers -----------------------------------------------------
    def emit(self, kind: str, payload) -> None:
        self.result_q.put((kind, payload))

    def stop_requested(self) -> bool:
        return self.stop_evt.is_set()

    def dump_params(self, params: dict) -> None:
        with open(self.params_path, 'w') as f:
            json.dump(params, f, indent=2)

    def save_checkpoint(self) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def base_params(self, cfg: dict, extra: dict | None = None) -> dict:
        """Common parameter record shared by all trainers."""
        dt = self.started_at
        params = {
            'date': {'year': dt.year, 'month': dt.month, 'day': dt.day,
                     'hour': dt.hour, 'minute': dt.minute, 'second': dt.second},
            'dropout_fc': cfg.get('dropout_fc'),
            'max_epochs': cfg.get('epochs'),
            'optimizer': {'name': 'Adam',
                          'kwargs': {'lr': cfg.get('lr'),
                                     'weight_decay': cfg.get('weight_decay')}},
            'architecture': str(self.model),
        }
        if extra:
            params.update(extra)
        return params

    # -- hooks (override per script) ---------------------------------------------
    def forward_batch(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prediction, target)`` for one loader item."""
        raise NotImplementedError

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce the raw criterion output to a scalar. Default: identity."""
        return loss

    # -- regression loop ---------------------------------------------------------
    def fit(self, train_loader, val_loader, epochs: int, log) -> None:
        """Regression epoch loop (loss + streaming R²), preserving prior behaviour."""
        for epoch in range(epochs):
            if self.stop_requested():
                self.emit('status', f'Stopped by user at epoch {epoch + 1}.')
                break

            t0 = time.time()
            self.model.train()
            train_loss = torch.tensor(0.0, device=self.device)
            for item in train_loader:
                out, labels = self.forward_batch(item)
                self.optimizer.zero_grad()
                loss = self.reduce_loss(self.criterion(out, labels))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.detach()
            epoch_train_loss = (train_loss / len(train_loader)).item()
            t1 = time.time()

            self.model.eval()
            r2 = R2Accumulator(self.n_outputs, device=self.device)
            val_loss_sum = 0.0
            with torch.no_grad():
                for item in val_loader:
                    out, labels = self.forward_batch(item)
                    val_loss_sum += self.reduce_loss(self.criterion(out, labels)).item()
                    r2.update(labels, out)
            r2_per_ch = r2.r2_per_channel().tolist()
            r2_mean = sum(r2_per_ch) / len(r2_per_ch)
            epoch_val_loss = val_loss_sum / len(val_loader)
            self.scheduler.step(epoch_val_loss)

            r2_cols = '\t'.join(f'{v:.6f}' for v in r2_per_ch)
            log.write(f'{epoch + 1}\t{epoch_train_loss:.6f}\t{epoch_val_loss:.6f}\t'
                      f'{r2_mean:.6f}\t{r2_cols}\t{self.optimizer.param_groups[0]["lr"]:.2e}\n')
            log.flush()

            self.emit('epoch', {
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'r2_mean': r2_mean,
                'r2_per_ch': r2_per_ch,
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': t1 - t0,
            })

    def test(self, test_loader) -> None:
        """Final unbiased test evaluation → results CSV + ``done`` event."""
        import pandas as pd
        from itertools import chain

        r2 = R2Accumulator(self.n_outputs, device=self.device)
        test_loss_sum = 0.0
        rows = []
        self.model.eval()
        with torch.no_grad():
            for item in test_loader:
                out, labels = self.forward_batch(item)
                rows.extend(zip(labels.cpu().tolist(), out.cpu().tolist()))
                test_loss_sum += self.reduce_loss(self.criterion(out, labels)).item()
                r2.update(labels, out)

        cols = ([f'true_{i}' for i in range(self.n_outputs)] +
                [f'pred_{i}' for i in range(self.n_outputs)])
        flat = [list(chain(t, p)) for t, p in rows]
        pd.DataFrame(flat, columns=cols).to_csv(self.results_path, index=False)

        r2_per_joint = r2.r2_per_channel()
        self.emit('done', {
            'r2_per_joint': r2_per_joint.tolist(),
            'r2_mean': r2_per_joint.mean().item(),
            'test_loss': test_loss_sum / len(test_loader),
            'run_id': self.run_id,
        })

    def log_header(self, log) -> None:
        """Write the regression training-log header."""
        r2_header = '\t'.join(f'Val R2[{i}]' for i in range(self.n_outputs))
        log.write(f'Epoch\tTrain Loss\tVal Loss\tVal R2 Mean\t{r2_header}\tLR\n')
