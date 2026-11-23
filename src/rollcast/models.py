"""Forecasters and the registry that holds them.

A forecaster is a function ``(history, horizon, season) -> forecast``. It is
handed a read-only copy of the past and nothing else, so it cannot see the
future. Add your own with the ``@register("name")`` decorator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Forecaster = Callable[[np.ndarray, int, int], np.ndarray]
MODELS: dict[str, Forecaster] = {}

SES_ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
RIDGE = 1.0


def register(name: str) -> Callable[[Forecaster], Forecaster]:
    def wrap(fn: Forecaster) -> Forecaster:
        if name in MODELS:
            raise ValueError(f"model {name!r} is already registered")
        MODELS[name] = fn
        return fn

    return wrap


@register("naive")
def naive(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    return np.full(horizon, history[-1])


@register("seasonal-naive")
def seasonal_naive(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    if season < 2 or len(history) < season:
        return naive(history, horizon, season)
    last_cycle = history[-season:]
    return np.array([last_cycle[k % season] for k in range(horizon)])


@register("drift")
def drift(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    slope = (history[-1] - history[0]) / max(len(history) - 1, 1)
    return history[-1] + slope * np.arange(1, horizon + 1)


@register("window-mean")
def window_mean(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    window = max(season, 2)
    return np.full(horizon, history[-window:].mean())


def _seasonal_profile(history: np.ndarray, season: int) -> np.ndarray:
    """Average shape of one season, with the level removed."""
    cycles = len(history) // season
    block = history[len(history) - cycles * season :].reshape(cycles, season)
    profile = (block - block.mean(axis=1, keepdims=True)).mean(axis=0)
    # Phase 0 of the profile is the first index of the trimmed block.
    shift = (len(history) - cycles * season) % season
    return np.roll(profile, shift)


@register("ets-lite")
def ets_lite(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Additive seasonal profile plus simple exponential smoothing of the rest."""
    if season < 2 or len(history) < 2 * season:
        return naive(history, horizon, season)
    profile = _seasonal_profile(history, season)
    phase = np.arange(len(history)) % season
    deseasonal = history - profile[phase]

    best_alpha, best_sse = SES_ALPHAS[0], np.inf
    for alpha in SES_ALPHAS:
        level, sse = deseasonal[0], 0.0
        for value in deseasonal[1:]:
            sse += (value - level) ** 2
            level = alpha * value + (1 - alpha) * level
        if sse < best_sse:
            best_alpha, best_sse = alpha, sse

    level = deseasonal[0]
    for value in deseasonal[1:]:
        level = best_alpha * value + (1 - best_alpha) * level
    future_phase = (len(history) + np.arange(horizon)) % season
    return level + profile[future_phase]


@register("ar-ridge")
def ar_ridge(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Linear autoregression on the last ``2 * season`` values, fitted by ridge."""
    lags = min(max(2 * season, 4), len(history) // 3)
    if lags < 2:
        return naive(history, horizon, season)
    mean, std = history.mean(), history.std() or 1.0
    z = (history - mean) / std
    rows = np.stack([z[i : i + lags] for i in range(len(z) - lags)])
    design = np.hstack([rows, np.ones((len(rows), 1))])
    target = z[lags:]
    gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ target)

    window = list(z[-lags:])
    out = []
    for _ in range(horizon):
        step = float(np.dot(weights[:-1], window[-lags:]) + weights[-1])
        out.append(step)
        window.append(step)
    return np.array(out) * std + mean
