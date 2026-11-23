"""Series to benchmark on: a synthetic panel, or your own CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def synthetic_panel(
    n_series: int = 6, length: int = 600, season: int = 24, seed: int = 0
) -> dict[str, np.ndarray]:
    """Seasonal series with a trend, autocorrelated noise and, for every other
    series, a level shift at 70% of the way through (a regime change)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    panel: dict[str, np.ndarray] = {}
    for i in range(n_series):
        level = rng.uniform(50, 150)
        trend = rng.uniform(-0.02, 0.05)
        amp = rng.uniform(5, 25)
        phase = rng.uniform(0, 2 * np.pi)
        wave = amp * np.sin(2 * np.pi * t / season + phase)
        wave += 0.3 * amp * np.sin(4 * np.pi * t / season)
        noise = np.zeros(length)
        eps = rng.normal(0, amp * 0.15, length)
        for k in range(1, length):
            noise[k] = 0.6 * noise[k - 1] + eps[k]
        series = level + trend * t + wave + noise
        if i % 2 == 1:
            series[int(0.7 * length) :] += rng.choice([-1, 1]) * amp * 1.5
        panel[f"series-{i}"] = series
    return panel


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """One column per series, a header row of names, no index column."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        names = next(reader, None)
        if not names:
            raise ValueError(f"{path} is empty")
        if len(set(names)) != len(names):
            raise ValueError(f"{path} has duplicate column names")
        columns: list[list[float]] = [[] for _ in names]
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue  # blank lines, such as a trailing newline
            if len(row) != len(names):
                raise ValueError(f"{path} line {line_no}: expected {len(names)} columns")
            for name, col, cell in zip(names, columns, row, strict=True):
                try:
                    col.append(float(cell))
                except ValueError:
                    raise ValueError(f"{path} line {line_no}, column {name!r}: {cell!r}") from None
    panel = {name: np.asarray(col, dtype=float) for name, col in zip(names, columns, strict=True)}
    if any(len(v) == 0 or not np.isfinite(v).all() for v in panel.values()):
        raise ValueError(f"{path} must contain finite numbers")
    return panel
