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

Options:
  --sir              Use SIR model instead of SEIR (default: SEIR)
  --plots            Save plots (default: enabled)
  --no-plots         Do not save plots
  --seasonal_amp     Amplitude for seasonality (e.g. 0.3)
  --seasonal_period  Period for seasonality in days (e.g. 120)
  --seasonal_phase   Phase offset for seasonality (e.g. 0)
  --wave_ramps       Comma-separated ramp widths for each wave (e.g. 7,10)
  --wave_durations   Comma-separated durations for each wave (e.g. 40,60)
  --help             Show this help message and exit

Examples:
  python main.py interact --seasonal_amp 0.4 --seasonal_period 120
  python main.py batch 10 --wave_ramps 7,10 --wave_durations 40,60 --sir --no-plots
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
    parser.add_argument('--sir', action='store_true', help='Use SIR model instead of SEIR')
    parser.add_argument('--plots', dest='save_plots', action='store_true', help='Save plots (default)')
    parser.add_argument('--no-plots', dest='save_plots', action='store_false', help='Do not save plots')
    # Batch fixed params
    parser.add_argument('--days', type=int, help='Number of days for each dataset in batch')
    parser.add_argument('--mask_score', type=int, help='Mask adherence 1-10')
    parser.add_argument('--multi_wave_count', type=int, help='Number of multi-waves/variants')
    parser.add_argument('--wave_days', type=str, help='Comma-separated days for each wave (e.g. 60,100)')
    parser.add_argument('--wave_betas', type=str, help='Comma-separated beta multipliers for each wave (e.g. 2.5,2.0)')
    parser.add_argument('--wave_seeds', type=str, help='Comma-separated seeds for each wave (e.g. 100,80)')
    # Advanced seasonality/multi-wave
    parser.add_argument('--seasonal_amp', type=float, help='Amplitude for seasonality (e.g. 0.3)')
    parser.add_argument('--seasonal_period', type=int, help='Period for seasonality in days (e.g. 120)')
    parser.add_argument('--seasonal_phase', type=float, help='Phase offset for seasonality (e.g. 0)')
    parser.add_argument('--wave_ramps', type=str, help='Comma-separated ramp widths for each wave (e.g. 7,10)')
    parser.add_argument('--wave_durations', type=str, help='Comma-separated durations for each wave (e.g. 40,60)')
    parser.set_defaults(save_plots=True)
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Warning: Unknown arguments: {unknown}")

    # Prepare batch_fixed dict, skipping None values
    batch_fixed = {k: v for k, v in {
        'days': args.days,
        'mask_score': args.mask_score,
        'multi_wave_count': args.multi_wave_count,
        'wave_days': args.wave_days,
        'wave_betas': args.wave_betas,
        'wave_seeds': args.wave_seeds,
        'seasonal_amp': args.seasonal_amp,
        'seasonal_period': args.seasonal_period,
        'seasonal_phase': args.seasonal_phase,
        'wave_ramps': args.wave_ramps,
        'wave_durations': args.wave_durations
    }.items() if v is not None}

    if args.mode == 'interact':
        params = get_user_inputs()
        run_single(params, use_sir=args.sir, save_plots=args.save_plots, cli_args=args)
    elif args.mode == 'batch':
        generate_batch(args.count, use_sir=args.sir, save_plots=args.save_plots, batch_fixed=batch_fixed)
    elif args.mode is None:
        params = generate_random_inputs()
        run_single(params, use_sir=args.sir, save_plots=args.save_plots, cli_args=args)
    else:
        print("Unknown mode! Use: 'interact', 'batch <count>', or --help.") 