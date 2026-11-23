import numpy as np
import pytest

from rollcast.backtest import forecast_at, mase_scale, run_backtest
from rollcast.cli import format_report, main
from rollcast.data import read_csv, synthetic_panel
from rollcast.models import MODELS, register

SEASON = 12


def periodic(n: int = 120) -> np.ndarray:
    return 10 + np.sin(2 * np.pi * np.arange(n) / SEASON)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_returns_finite_forecast_of_right_length(name: str) -> None:
    y = synthetic_panel(1, 200, SEASON)["series-0"]
    out = MODELS[name](y, 7, SEASON)
    assert out.shape == (7,) and np.isfinite(out).all()


def test_seasonal_naive_is_exact_on_a_periodic_series() -> None:
    y = periodic()
    pred = MODELS["seasonal-naive"](y[:100], 20, SEASON)
    assert np.allclose(pred, y[100:120])


def test_drift_is_exact_on_a_line() -> None:
    y = 3.0 + 2.0 * np.arange(50)
    assert np.allclose(MODELS["drift"](y[:40], 10, SEASON), y[40:50])


def test_ets_lite_and_ar_ridge_handle_a_clean_seasonal_series() -> None:
    y = periodic()
    for name in ("ets-lite", "ar-ridge"):
        pred = MODELS[name](y[:100], 12, SEASON)
        assert np.abs(pred - y[100:112]).mean() < 0.2, name


def test_short_history_falls_back_instead_of_crashing() -> None:
    short = np.array([1.0, 2.0, 3.0])
    for name in ("seasonal-naive", "ets-lite", "ar-ridge"):
        assert np.isfinite(MODELS[name](short, 4, SEASON)).all()


def test_forecasts_cannot_see_the_future() -> None:
    """Changing values after the cutoff must not change any forecast."""
    y = synthetic_panel(1, 200, SEASON)["series-0"]
    tampered = y.copy()
    tampered[120:] += 1000.0
    for name, model in MODELS.items():
        a = forecast_at(model, y, 120, 12, SEASON)
        b = forecast_at(model, tampered, 120, 12, SEASON)
        assert np.array_equal(a, b), f"{name} used data after the cutoff"


def test_history_is_read_only() -> None:
    def vandal(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
        history[0] = 0.0
        return np.zeros(horizon)

    with pytest.raises(ValueError):
        forecast_at(vandal, periodic(), 60, 4, SEASON)


def test_bad_model_output_is_rejected() -> None:
    with pytest.raises(ValueError):
        forecast_at(lambda h, n, s: np.zeros(n + 1), periodic(), 60, 4, SEASON)
    with pytest.raises(ValueError):
        forecast_at(lambda h, n, s: np.full(n, np.nan), periodic(), 60, 4, SEASON)


def test_mase_scale_matches_seasonal_naive_error() -> None:
    assert mase_scale(periodic(), SEASON) == pytest.approx(0.0, abs=1e-9)
    assert mase_scale(np.array([1.0, 3.0, 2.0]), 1) == pytest.approx(1.5)


def test_backtest_ranks_the_matching_model_first() -> None:
    noise = np.random.default_rng(0).normal(0, 0.02, 200)
    panel = {"p": periodic(200) + noise}
    names = ("naive", "seasonal-naive")
    bt = run_backtest(panel, horizon=12, season=SEASON, step=12, model_names=names)
    board = bt.leaderboard()
    assert board[0][0] == "seasonal-naive" and board[0][1] < 2.0 < board[1][1]


def test_a_flat_history_is_skipped_instead_of_exploding_the_score() -> None:
    flat_then_ramp = np.concatenate([np.full(100, 5.0), 5.0 + np.arange(100)])
    bt = run_backtest(
        {"flat": flat_then_ramp, "ok": periodic(200)},
        horizon=4,
        season=4,
        start=0.3,
        step=8,
        model_names=("naive", "seasonal-naive"),
    )
    assert np.isnan(bt.mase["naive"]).any(), "flat-history origins should be masked"
    assert all(v < 1e3 for v in bt.mean_mase().values())
    assert len(bt.running_leader()) == len(bt.cutoffs)


def test_all_flat_series_raise() -> None:
    with pytest.raises(ValueError):
        run_backtest({"c": np.full(200, 3.0)}, horizon=4, season=4)


def test_run_backtest_rejects_nonsense_settings() -> None:
    for kwargs in ({"step": 0}, {"horizon": 0}, {"season": 0}, {"start": 1.5}):
        with pytest.raises(ValueError):
            run_backtest({"a": periodic(200)}, **kwargs)


def test_csv_tolerates_blank_lines_and_bom_and_reports_bad_cells(tmp_path) -> None:
    ok = tmp_path / "ok.csv"
    ok.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n\n3,4\n\n")
    assert read_csv(ok)["a"].tolist() == [1.0, 3.0]
    dup = tmp_path / "dup.csv"
    dup.write_text("a,a\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_csv(dup)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2, column 'b'"):
        read_csv(bad)


def test_cli_reports_errors_without_a_traceback(tmp_path, capsys) -> None:
    assert main(["run", str(tmp_path / "missing.csv")]) == 1
    assert "rollcast:" in capsys.readouterr().err


def test_backtest_validates_inputs() -> None:
    with pytest.raises(ValueError):
        run_backtest({})
    with pytest.raises(ValueError):
        run_backtest({"a": periodic()}, model_names=("nope",))
    with pytest.raises(ValueError):
        run_backtest({"a": periodic(40)}, horizon=30, season=SEASON)


def test_registry_accepts_new_models_and_rejects_duplicates() -> None:
    @register("test-zero")
    def zero(history: np.ndarray, horizon: int, season: int) -> np.ndarray:
        return np.zeros(horizon)

    try:
        assert "test-zero" in MODELS
        with pytest.raises(ValueError):
            register("test-zero")(zero)
    finally:
        MODELS.pop("test-zero")


def test_running_leader_has_one_entry_per_origin() -> None:
    bt = run_backtest(synthetic_panel(2, 300, SEASON), horizon=12, season=SEASON, step=24)
    assert len(bt.running_leader()) == len(bt.cutoffs)


def test_csv_round_trip_and_errors(tmp_path) -> None:
    good = tmp_path / "ok.csv"
    good.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    panel = read_csv(good)
    assert list(panel) == ["a", "b"] and panel["b"].tolist() == [2.0, 4.0]
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(bad)
    nan = tmp_path / "nan.csv"
    nan.write_text("a\nnan\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(nan)


def test_cli_run_on_csv(tmp_path, capsys) -> None:
    panel = synthetic_panel(2, 200, SEASON)
    path = tmp_path / "d.csv"
    rows = ["series-0,series-1"] + [f"{a},{b}" for a, b in zip(*panel.values(), strict=True)]
    path.write_text("\n".join(rows), encoding="utf-8")
    assert main(["run", str(path), "--horizon", "12", "--season", str(SEASON)]) == 0
    assert "seasonal-naive" in capsys.readouterr().out


def test_report_mentions_leader_changes() -> None:
    bt = run_backtest(synthetic_panel(2, 300, SEASON), horizon=12, season=SEASON, step=24)
    assert "leader as data arrived" in format_report(bt, 12)
