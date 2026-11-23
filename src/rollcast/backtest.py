"""Rolling-origin evaluation: the benchmark loop and its scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rollcast.models import MODELS, Forecaster

MIN_SCALE = 1e-9  # a history flatter than this has no meaningful MASE scale


@dataclass(frozen=True)
class Backtest:
    models: tuple[str, ...]
    series: tuple[str, ...]
    cutoffs: tuple[int, ...]
    # mase[model] has shape (n_series, n_cutoffs)
    mase: dict[str, np.ndarray]

    def mean_mase(self) -> dict[str, float]:
        """Mean over every (series, origin) with a usable scale; flat histories are NaN."""
        return {m: float(np.nanmean(self.mase[m])) for m in self.models}

    def leaderboard(self) -> list[tuple[str, float, float]]:
        """(model, mean MASE, skill vs seasonal-naive), best first."""
        means = self.mean_mase()
        base = means.get("seasonal-naive")
        rows = [(m, v, (1 - v / base) if base else float("nan")) for m, v in means.items()]
        return sorted(rows, key=lambda r: r[1])

    def running_leader(self) -> list[str]:
        """Best model after each cutoff, using every cutoff seen so far."""
        stack = np.stack([self.mase[m] for m in self.models])  # (model, series, cutoff)
        sums = np.cumsum(np.nansum(stack, axis=1), axis=1)
        counts = np.cumsum((~np.isnan(stack)).sum(axis=1), axis=1)
        average = sums / np.maximum(counts, 1)
        return [self.models[i] for i in average.argmin(axis=0)]


def mase_scale(history: np.ndarray, season: int) -> float:
    """Mean absolute error of the seasonal-naive rule inside ``history``."""
    lag = season if season >= 2 and len(history) > season else 1
    return float(np.abs(history[lag:] - history[:-lag]).mean())


def forecast_at(
    model: Forecaster, y: np.ndarray, cutoff: int, horizon: int, season: int
) -> np.ndarray:
    """Forecast from ``y[:cutoff]`` only. The slice is a read-only copy."""
    history = y[:cutoff].copy()
    history.setflags(write=False)
    out = np.asarray(model(history, horizon, season), dtype=float)
    if out.shape != (horizon,) or not np.isfinite(out).all():
        raise ValueError(f"model returned an invalid forecast of shape {out.shape}")
    return out


def run_backtest(
    panel: dict[str, np.ndarray],
    horizon: int = 24,
    season: int = 24,
    start: float = 0.5,
    step: int = 12,
    model_names: tuple[str, ...] | None = None,
) -> Backtest:
    if not panel:
        raise ValueError("panel is empty")
    if horizon < 1 or season < 1 or step < 1:
        raise ValueError("horizon, season and step must all be at least 1")
    if not 0.0 < start < 1.0:
        raise ValueError("start must be between 0 and 1")
    names = model_names or tuple(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    length = min(len(v) for v in panel.values())
    first = max(int(length * start), 2 * season + 2)
    cutoffs = tuple(range(first, length - horizon + 1, step))
    if not cutoffs:
        raise ValueError("series are too short for this horizon and start")

    mase = {m: np.zeros((len(panel), len(cutoffs))) for m in names}
    for si, y in enumerate(panel.values()):
        for ci, cut in enumerate(cutoffs):
            actual = y[cut : cut + horizon]
            scale = mase_scale(y[:cut], season)
            for m in names:
                pred = forecast_at(MODELS[m], y, cut, horizon, season)
                error = np.abs(actual - pred).mean()
                mase[m][si, ci] = error / scale if scale > MIN_SCALE else np.nan
    if all(np.isnan(v).all() for v in mase.values()):
        raise ValueError("every history is flat, so MASE is undefined")
    return Backtest(tuple(names), tuple(panel), cutoffs, mase)
