"""Shared NiceGUI helpers for the training front-ends.

The three ``train_*_ui`` scripts used identical chart-option builders and
config load/save helpers; they live here so the scripts import rather than
duplicate them.
"""

from __future__ import annotations

import json
import os


def chart_options(title: str, series_names: list[str], y_label: str = '') -> dict:
    """ECharts option dict for a smooth multi-series line chart keyed by epoch."""
    return {
        'title': {'text': title, 'textStyle': {'fontSize': 13}},
        'tooltip': {'trigger': 'axis'},
        'legend': {'data': series_names, 'bottom': 0},
        'grid': {'left': '12%', 'right': '4%', 'top': '18%', 'bottom': '18%'},
        'xAxis': {'type': 'category', 'name': 'Epoch', 'data': []},
        'yAxis': {'type': 'value', 'name': y_label},
        'series': [{'name': n, 'type': 'line', 'data': [], 'smooth': True,
                    'showSymbol': False} for n in series_names],
    }


class ConfigStore:
    """Load/save the dataset-path config JSON, backed by a set of defaults."""

    def __init__(self, config_file: str, defaults: dict):
        self.config_file = config_file
        self.defaults = defaults

    def load(self) -> dict:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file) as f:
                    return {**self.defaults, **json.load(f)}
            except Exception:
                pass
        return self.defaults.copy()

    def save(self, paths: dict) -> None:
        with open(self.config_file, 'w') as f:
            json.dump(paths, f, indent=2)
