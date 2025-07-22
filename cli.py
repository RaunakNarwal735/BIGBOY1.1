"""
- **Dataset ALWAYS saved** (CSV + metadata) regardless of `--plots` selection.
- `--plots all` now: writes dataset via `save_outputs(..., save_plots=[])` (no base plots),
  then generates the full advanced suite via `advanced_plotting.save_all_advanced_plots()`.
- `--plots sir` expands to the classic two plots (`sir`, `reported`).
- Any comma list (e.g., `sir,reported,heatmap`) passed through to `save_outputs()`.
- Seasonality/wave overrides added & safely applied (no AttributeError).

Modes:
    interact  -> prompt user for params
    batch N   -> N random runs (or partially fixed via args)
    (blank)   -> single random run ("random mode")

Usage examples:
    python BIGBOY1.2.py --plots all
    python BIGBOY1.2.py --plots sir
    python BIGBOY1.2.py batch 10 --plots sir
    python BIGBOY1.2.py interact --plots all
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Union

from config import BASE_SAVE_DIR
from input_handlers import get_user_inputs, generate_random_inputs
from simulation import simulate_epidemic
from output import save_outputs

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def ts() -> str:
    """Timestamp string for unique output directories."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cli_error(msg: str) -> None:
    print(f"Error: {msg}")
    raise SystemExit(1)

# Plot selection parsing


PlotSelection = Union[str, List[str]]  # 'all' OR list like ['sir','reported']


def parse_plot_arg(raw: str | None) -> PlotSelection:
    """Normalize the --plots argument.

    Returns:
        'all'              -> user requested all advanced plots.
        ['sir','reported'] -> user requested sir (classic minimal set).
        [...]              -> cleaned list of user tokens.
        []                 -> no plots.
    """
    if not raw:
        return []
    tokens = [t.strip().lower() for t in raw.split(',') if t.strip()]
    if not tokens:
        return []
    if 'all' in tokens:
        return 'all'
    if tokens == ['sir']:
        return ['sir', 'reported']
    # Expand bare 'sir' among others? keep user intent: add reported if missing.
    if 'sir' in tokens and 'reported' not in tokens:
        tokens.append('reported')
    return tokens
# Seasonality / wave arg utilities

def _parse_csv_floats(s: str | None) -> List[float] | None:
    if s is None:
        return None
    out: List[float] = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            print(f"Warning: could not parse float in list: {tok}")
    return out if out else None


def _parse_csv_ints(s: str | None) -> List[int] | None:
    vals = _parse_csv_floats(s)
    if vals is None:
        return None
    return [int(round(v)) for v in vals]

# Core run functions

def _apply_cli_overrides(params: Dict[str, Any], cli_args: argparse.Namespace) -> None:
    """Inject optional advanced params into params dict (in place)."""
    # Seasonality overrides
    if getattr(cli_args, 'seasonal_amp', None) is not None:
        params['seasonal_amp'] = cli_args.seasonal_amp
    if getattr(cli_args, 'seasonal_period', None) is not None:
        params['seasonal_period'] = cli_args.seasonal_period
    if getattr(cli_args, 'seasonal_phase', None) is not None:
        params['seasonal_phase'] = cli_args.seasonal_phase

    # Multi-wave shape overrides (if model supports)
    ramps = _parse_csv_floats(getattr(cli_args, 'wave_ramps', None))
    durs = _parse_csv_ints(getattr(cli_args, 'wave_durations', None))
    if 'waves' in params and isinstance(params['waves'], list):
        if ramps:
            for j, wave in enumerate(params['waves']):
                if j < len(ramps):
                    wave['ramp'] = ramps[j]
        if durs:
            for j, wave in enumerate(params['waves']):
                if j < len(durs):
                    wave['duration'] = durs[j]


def _do_save_plots(plot_sel: PlotSelection, df, params, out_dir: str) -> None:
    """ALWAYS write dataset; then generate plots per selection."""
    # 1. Always save dataset (CSV, metadata) FIRST. Force no basic plots here.
    #    This guarantees the user always gets a dataset even in --plots all mode.
    save_outputs(df, params, out_dir, save_plots=[])  # dataset only

    # 2. Now handle plot selection
    if plot_sel == 'all':
        try:
            import numpy as np  # local
            from plotting import save_all_advanced_plots
        except Exception as e:  # advanced plotting missing
            print(f"[WARN] advanced_plotting not available ({e}); falling back to classic SIR plots.")
            # generate classic 2 plots
            save_outputs(df, params, out_dir, save_plots=['sir', 'reported'])
        else:
            save_all_advanced_plots(df, out_dir)
        return

    # Non-all: user wanted specific basic plots (sir/report/...). Use save_outputs.
    if isinstance(plot_sel, list) and plot_sel:
        save_outputs(df, params, out_dir, save_plots=plot_sel)
    # If user passed empty list, nothing extra (dataset already saved).


def run_single(params: Dict[str, Any], use_sir: bool, plot_sel: PlotSelection, cli_args: argparse.Namespace | None = None) -> None:
    """Run one simulation and save outputs/plots."""
    if cli_args is not None:
        _apply_cli_overrides(params, cli_args)
    df = simulate_epidemic(params, use_sir=use_sir)
    out_dir = os.path.join(BASE_SAVE_DIR, f"run_{ts()}")
    _do_save_plots(plot_sel, df, params, out_dir)
    print(f"Run saved in: {out_dir}")


def generate_batch(count: int = 5, use_sir: bool = False, plot_sel: PlotSelection = 'all', batch_fixed: Dict[str, Any] | None = None, cli_args: argparse.Namespace | None = None) -> None:
    """Generate many random datasets and save each set of plots."""
    if count < 1:
        _cli_error("Batch count must be at least 1.")
    batch_id = ts()
    base_batch_dir = os.path.join(BASE_SAVE_DIR, f"batch_{batch_id}")
    print(f"\n=== BATCH GENERATION: {count} datasets ===")
    for i in range(1, count + 1):
        params = generate_random_inputs(batch_fixed=batch_fixed)
        if cli_args is not None:
            _apply_cli_overrides(params, cli_args)
        df = simulate_epidemic(params, use_sir=use_sir)
        run_dir = os.path.join(base_batch_dir, f"run_{i:03d}")
        _do_save_plots(plot_sel, df, params, run_dir)
        print(f"  Dataset {i}/{count} -> {os.path.join(run_dir, 'dataset.csv')}")
    print(f"\nAll datasets saved in: {base_batch_dir}")

# Argument parser

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Epidemic Dataset Generator CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('mode', nargs='?', default=None, choices=['interact', 'batch', None],
                        help="Mode: interact (prompt), batch (generate many), or blank for single run")
    parser.add_argument('count', nargs='?', type=int, default=5, help="Number of datasets when in batch mode")

    # Core user parameters (all optional; used to override random params)
    parser.add_argument('--days', type=int, help='Number of days')
    parser.add_argument('--mask_score', type=int, help='Mask adherence 1-10')
    parser.add_argument('--crowdedness_score', type=int, help='Crowdedness 1-10')
    parser.add_argument('--quarantine_enabled', type=str, help="Enable quarantine (y/n)")
    parser.add_argument('--vaccination_enabled', type=str, help="Enable vaccination (y/n)")
    parser.add_argument('--travel_enabled', type=str, help="Enable travel (y/n)")
    parser.add_argument('--travel_max', type=int, help='Max imported cases per day')
    parser.add_argument('--incubation_period', type=int, help='Incubation period (days)')
    parser.add_argument('--daily_vaccination_rate', type=float, help='Fraction vaccinated per day (0-1)')
    parser.add_argument('--initial_infected', type=int, help='Initial infected count')
    parser.add_argument('--population', type=int, help='Population size')
    parser.add_argument('--testing_rate', type=str, help='Testing rate (low/medium/high)')
    parser.add_argument('--multi_wave', type=str, help='Enable multi-wave (y/n)')
    parser.add_argument('--mask_decay_rate', type=float, help='Mask decay rate per day (fraction)')
    parser.add_argument('--reporting_prob_min', type=float, help='Min reporting probability (0-1)')
    parser.add_argument('--reporting_prob_max', type=float, help='Max reporting probability (0-1)')

    # Seasonality / transmission overrides (safe optional)
    parser.add_argument('--seasonal_amp', type=float, default=None,
                        help='Amplitude for seasonal forcing of transmission')
    parser.add_argument('--seasonal_period', type=float, default=None,
                        help='Season length in days (e.g., 365)')
    parser.add_argument('--seasonal_phase', type=float, default=None,
                        help='Phase shift in days')

    # Multi-wave shape overrides
    parser.add_argument('--wave_ramps', type=str, default=None,
                        help='Comma-separated multipliers per wave (e.g., 1.0,1.5,0.8)')
    parser.add_argument('--wave_durations', type=str, default=None,
                        help='Comma-separated durations per wave (days)')

    # Plot selection
    parser.add_argument('--plots', type=str, default=None,
                        help="Comma-separated plots to save. Use 'sir' for classic 2 plots, 'all' for full advanced suite.")

    return parser
# Validation helpers

def _validate_args(args: argparse.Namespace) -> None:
    if args.mask_score is not None and not (1 <= args.mask_score <= 10):
        _cli_error("mask_score must be between 1 and 10")
    if args.crowdedness_score is not None and not (1 <= args.crowdedness_score <= 10):
        _cli_error("crowdedness_score must be between 1 and 10")
    if args.days is not None and args.days <= 0:
        _cli_error("days must be positive")
    if args.population is not None and args.population <= 0:
        _cli_error("population must be positive")

    for flag in ['quarantine_enabled', 'vaccination_enabled', 'travel_enabled', 'multi_wave']:
        val = getattr(args, flag, None)
        if val is not None and val not in ('y', 'n'):
            _cli_error(f"{flag} must be 'y' or 'n'")

    if args.testing_rate is not None and args.testing_rate not in ('low', 'medium', 'high'):
        _cli_error("testing_rate must be 'low', 'medium', or 'high'")

    if args.daily_vaccination_rate is not None and not (0 <= args.daily_vaccination_rate <= 1):
        _cli_error("daily_vaccination_rate must be 0-1")

    if args.incubation_period is not None and args.incubation_period <= 0:
        _cli_error("incubation_period must be positive")

    if args.travel_max is not None and args.travel_max < 0:
        _cli_error("travel_max must be >= 0")

    if args.reporting_prob_min is not None and not (0 <= args.reporting_prob_min <= 1):
        _cli_error("reporting_prob_min must be 0-1")
    if args.reporting_prob_max is not None and not (0 <= args.reporting_prob_max <= 1):
        _cli_error("reporting_prob_max must be 0-1")
    if (args.reporting_prob_min is not None and args.reporting_prob_max is not None and
            args.reporting_prob_max < args.reporting_prob_min):
        _cli_error("reporting_prob_max must be >= reporting_prob_min")


# ------------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Warning: Unknown arguments ignored: {unknown}")

    _validate_args(args)

    # Build dict of user overrides (only non-None)
    batch_fixed = {k: v for k, v in {
        'days': args.days,
        'mask_score': args.mask_score,
        'crowdedness_score': args.crowdedness_score,
        'quarantine_enabled': args.quarantine_enabled,
        'vaccination_enabled': args.vaccination_enabled,
        'travel_enabled': args.travel_enabled,
        'travel_max': args.travel_max,
        'incubation_period': args.incubation_period,
        'daily_vaccination_rate': args.daily_vaccination_rate,
        'initial_infected': args.initial_infected,
        'population': args.population,
        'testing_rate': args.testing_rate,
        'multi_wave': args.multi_wave,
        'mask_decay_rate': args.mask_decay_rate,
        'reporting_prob_min': args.reporting_prob_min,
        'reporting_prob_max': args.reporting_prob_max,
    }.items() if v is not None}

    # Parse plots selection
    plot_sel = parse_plot_arg(args.plots)

    if args.mode == 'interact':
        params = get_user_inputs()
        run_single(params, use_sir=False, plot_sel=plot_sel, cli_args=args)
        return

    if args.mode == 'batch':
        generate_batch(args.count, use_sir=False, plot_sel=plot_sel, batch_fixed=batch_fixed, cli_args=args)
        return

    # Mode None -> random single (with overrides if provided)
    params = generate_random_inputs(batch_fixed=batch_fixed)
    run_single(params, use_sir=False, plot_sel=plot_sel, cli_args=args)


if __name__ == "__main__":  # pragma: no cover
    main()
