"""
CLI module for Epidemic Dataset Generator

Handles argument parsing, mode selection, batch generation, and help text.
Now supports advanced seasonality and multi-wave options.
"""

import argparse
import os
from datetime import datetime
from config import BASE_SAVE_DIR
from input_handlers import get_user_inputs, generate_random_inputs
from simulation import simulate_epidemic
from output import save_outputs


def ts():
    """
    Return a timestamp string for unique output directories.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_help():
    """
    Print CLI usage help text (not bound to --help, for reference only).
    """
    print("""
Epidemic Dataset Generator CLI Usage:

python main.py [mode] [options]

Modes:
  interact           Interactive mode (prompts for parameters)
  batch [N]          Generate a batch of N random datasets (default N=5)

Options (for batch or single run):
  --days             Number of days
  --mask_score       Mask adherence 1-10
  --crowdedness_score Crowdedness 1-10
  --quarantine_enabled y/n
  --vaccination_enabled y/n
  --travel_enabled y/n
  --travel_max       Max imported cases per day
  --incubation_period Incubation period (days)
  --daily_vaccination_rate Fraction vaccinated per day (e.g., 0.01)
  --initial_infected Initial infected
  --population       Population size
  --testing_rate     low/medium/high
  --multi_wave       y/n
  --mask_decay_rate  Mask decay rate per day (fraction)
  --reporting_prob_min Min reporting probability
  --reporting_prob_max Max reporting probability
  --plots            Comma-separated list of plots to save (e.g. sir,reported,rt)

Examples:
  python main.py batch 10 --days 120 --mask_score 3 --crowdedness_score 8 --quarantine_enabled y --plots sir,reported
  python main.py --days 120 --mask_score 3 --crowdedness_score 8 --quarantine_enabled y --plots sir,reported
  python main.py interact
  python main.py --help
""")


def generate_batch(count=5, use_sir=False, save_plots=True, batch_fixed=None):
    """
    Generate a batch of random datasets and save them.
    """
    if count < 1:
        print("Batch count must be at least 1.")
        return
    batch_id = ts()
    base_batch_dir = os.path.join(BASE_SAVE_DIR, f"batch_{batch_id}")
    print(f"\n=== BATCH GENERATION: {count} datasets ===")
    for i in range(1, count + 1):
        params = generate_random_inputs(batch_fixed=batch_fixed)
        # Inject advanced seasonality/multi-wave params if present
        if batch_fixed:
            if 'seasonal_amp' in batch_fixed:
                params['seasonal_amp'] = batch_fixed['seasonal_amp']
            if 'seasonal_period' in batch_fixed:
                params['seasonal_period'] = batch_fixed['seasonal_period']
            if 'seasonal_phase' in batch_fixed:
                params['seasonal_phase'] = batch_fixed['seasonal_phase']
            # Multi-wave ramps/durations
            if 'wave_ramps' in batch_fixed and params.get('waves'):
                ramps = [float(x) for x in batch_fixed['wave_ramps'].split(',')]
                for j, wave in enumerate(params['waves']):
                    if j < len(ramps):
                        wave['ramp'] = ramps[j]
            if 'wave_durations' in batch_fixed and params.get('waves'):
                durations = [int(x) for x in batch_fixed['wave_durations'].split(',')]
                for j, wave in enumerate(params['waves']):
                    if j < len(durations):
                        wave['duration'] = durations[j]
        df = simulate_epidemic(params, use_sir=use_sir)
        run_dir = os.path.join(base_batch_dir, f"run_{i:03d}")
        save_outputs(df, params, run_dir, save_plots=save_plots)
        print(f"  Dataset {i}/{count} -> {os.path.join(run_dir, 'dataset.csv')}")
    print(f"\nAll datasets saved in: {base_batch_dir}")


def run_single(params, use_sir, save_plots, cli_args=None):
    """
    Run a single simulation and save outputs. Inject advanced params if provided.
    """
    if cli_args:
        if cli_args.seasonal_amp is not None:
            params['seasonal_amp'] = cli_args.seasonal_amp
        if cli_args.seasonal_period is not None:
            params['seasonal_period'] = cli_args.seasonal_period
        if cli_args.seasonal_phase is not None:
            params['seasonal_phase'] = cli_args.seasonal_phase
        if cli_args.wave_ramps and params.get('waves'):
            ramps = [float(x) for x in cli_args.wave_ramps.split(',')]
            for j, wave in enumerate(params['waves']):
                if j < len(ramps):
                    wave['ramp'] = ramps[j]
        if cli_args.wave_durations and params.get('waves'):
            durations = [int(x) for x in cli_args.wave_durations.split(',')]
            for j, wave in enumerate(params['waves']):
                if j < len(durations):
                    wave['duration'] = durations[j]
    df = simulate_epidemic(params, use_sir=use_sir)
    out_dir = os.path.join(BASE_SAVE_DIR, f"run_{ts()}")
    save_outputs(df, params, out_dir, save_plots=save_plots)


def main():
    """
    Main CLI entrypoint for the epidemic dataset generator.
    Parses arguments and dispatches to the appropriate mode.
    """
    parser = argparse.ArgumentParser(
        description="Epidemic Dataset Generator CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('mode', nargs='?', default=None, choices=['interact', 'batch', None],
                        help="Mode: interact (interactive), batch (batch generation), or leave blank for random single run")
    parser.add_argument('count', nargs='?', type=int, default=5, help="Number of datasets for batch mode")
    # User-specific/fixable params
    parser.add_argument('--days', type=int, help='Number of days')
    parser.add_argument('--mask_score', type=int, help='Mask adherence 1-10')
    parser.add_argument('--crowdedness_score', type=int, help='Crowdedness 1-10')
    parser.add_argument('--quarantine_enabled', type=str, help='Enable quarantine (y/n)')
    parser.add_argument('--vaccination_enabled', type=str, help='Enable vaccination (y/n)')
    parser.add_argument('--travel_enabled', type=str, help='Enable travel (y/n)')
    parser.add_argument('--travel_max', type=int, help='Max imported cases per day')
    parser.add_argument('--incubation_period', type=int, help='Incubation period (days)')
    parser.add_argument('--daily_vaccination_rate', type=float, help='Fraction vaccinated per day (e.g., 0.01)')
    parser.add_argument('--initial_infected', type=int, help='Initial infected')
    parser.add_argument('--population', type=int, help='Population size')
    parser.add_argument('--testing_rate', type=str, help='Testing rate (low/medium/high)')
    parser.add_argument('--multi_wave', type=str, help='Enable multi-wave (y/n)')
    parser.add_argument('--mask_decay_rate', type=float, help='Mask decay rate per day (fraction)')
    parser.add_argument('--reporting_prob_min', type=float, help='Min reporting probability')
    parser.add_argument('--reporting_prob_max', type=float, help='Max reporting probability')
    parser.add_argument('--plots', type=str, help='Comma-separated list of plots to save (e.g. sir,reported,rt)')
    parser.set_defaults(plots=None)
    args, unknown = parser.parse_known_args()

    # --- Validation for user-specific parameters ---
    def cli_error(msg):
        print(f"Error: {msg}")
        exit(1)
    if args.mask_score is not None and not (1 <= args.mask_score <= 10):
        cli_error("mask_score must be between 1 and 10")
    if args.crowdedness_score is not None and not (1 <= args.crowdedness_score <= 10):
        cli_error("crowdedness_score must be between 1 and 10")
    if args.days is not None and args.days <= 0:
        cli_error("days must be a positive integer")
    if args.population is not None and args.population <= 0:
        cli_error("population must be a positive integer")
    for flag in ['quarantine_enabled', 'vaccination_enabled', 'travel_enabled', 'multi_wave']:
        val = getattr(args, flag, None)
        if val is not None and val not in ('y', 'n'):
            cli_error(f"{flag} must be 'y' or 'n'")
    if args.testing_rate is not None and args.testing_rate not in ('low', 'medium', 'high'):
        cli_error("testing_rate must be 'low', 'medium', or 'high'")
    if args.daily_vaccination_rate is not None and not (0 <= args.daily_vaccination_rate <= 1):
        cli_error("daily_vaccination_rate must be between 0 and 1")
    if args.incubation_period is not None and args.incubation_period <= 0:
        cli_error("incubation_period must be a positive integer")
    if args.travel_max is not None and args.travel_max < 0:
        cli_error("travel_max must be >= 0")
    if args.reporting_prob_min is not None and not (0 <= args.reporting_prob_min <= 1):
        cli_error("reporting_prob_min must be between 0 and 1")
    if args.reporting_prob_max is not None and not (0 <= args.reporting_prob_max <= 1):
        cli_error("reporting_prob_max must be between 0 and 1")
    if (args.reporting_prob_min is not None and args.reporting_prob_max is not None and
        args.reporting_prob_max < args.reporting_prob_min):
        cli_error("reporting_prob_max must be >= reporting_prob_min")

    if unknown:
        print(f"Warning: Unknown arguments: {unknown}")

    # Prepare batch_fixed dict, skipping None values
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
        'reporting_prob_max': args.reporting_prob_max
    }.items() if v is not None}

    if args.mode == 'interact':
        params = get_user_inputs()
        plot_list = [p.strip() for p in args.plots.split(',')] if args.plots else []
        run_single(params, use_sir=False, save_plots=plot_list, cli_args=args)
    elif args.mode == 'batch':
        plot_list = [p.strip() for p in args.plots.split(',')] if args.plots else []
        generate_batch(args.count, use_sir=False, save_plots=plot_list, batch_fixed=batch_fixed)
    elif args.mode is None:
        # Single run with fixed params if provided, else random
        params = generate_random_inputs(batch_fixed=batch_fixed)
        plot_list = [p.strip() for p in args.plots.split(',')] if args.plots else []
        run_single(params, use_sir=False, save_plots=plot_list, cli_args=args)
    else:
        print("Unknown mode! Use: 'interact', 'batch <count>', or --help.") 