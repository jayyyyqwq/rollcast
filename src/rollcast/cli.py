"""Command line: ``rollcast demo`` and ``rollcast run data.csv``."""

from __future__ import annotations

import argparse
import sys
from itertools import groupby
from pathlib import Path

from rollcast.backtest import Backtest, run_backtest
from rollcast.data import read_csv, synthetic_panel


def format_report(bt: Backtest, horizon: int) -> str:
    lines = [
        f"{len(bt.series)} series, {len(bt.cutoffs)} forecast origins, horizon {horizon}",
        "",
        f"{'rank':>4}  {'model':<16} {'MASE':>7}  {'skill vs seasonal-naive':>24}",
    ]
    for rank, (model, score, skill) in enumerate(bt.leaderboard(), start=1):
        lines.append(f"{rank:>4}  {model:<16} {score:>7.3f}  {skill:>+23.0%}")

    leaders = bt.running_leader()
    runs = [(k, len(list(g))) for k, g in groupby(leaders)]
    lines += ["", f"leader as data arrived (changed {len(runs) - 1} times):"]
    lines.append("  " + " -> ".join(f"{name} ({n})" for name, n in runs))
    lines += ["", "MASE below 1 beats the seasonal-naive rule on its own training data."]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rollcast", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="benchmark on a synthetic panel")
    run = sub.add_parser("run", help="benchmark on your CSV (one column per series)")
    run.add_argument("csv", type=Path)
    run.add_argument("--horizon", type=int, default=24)
    run.add_argument("--season", type=int, default=24)
    run.add_argument("--step", type=int, default=12, help="readings between origins")
    run.add_argument("--start", type=float, default=0.5, help="share of data before origin 1")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as e:
        print(f"rollcast: {e}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    if args.command == "demo":
        bt = run_backtest(synthetic_panel())
        print(format_report(bt, 24))
        return 0
    bt = run_backtest(read_csv(args.csv), args.horizon, args.season, args.start, args.step)
    print(format_report(bt, args.horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
