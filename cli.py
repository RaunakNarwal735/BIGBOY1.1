"""
CLI module for Epidemic Dataset Generator

Handles argument parsing, mode selection, batch generation, and help text.
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
    Print CLI usage help text.
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
  --help             Show this help message and exit

Examples:
  python main.py interact
  python main.py batch 10 --sir --no-plots
  python main.py --help
""")


def generate_batch(count=5, use_sir=False, save_plots=True, batch_fixed=None):
    """
    Generate a batch of random datasets and save them.
    Args:
        count (int): Number of datasets to generate.
        use_sir (bool): Use SIR model if True.
        save_plots (bool): Whether to save plots.
        batch_fixed (dict): Fixed parameters for batch (optional).
    """
    batch_id = ts()
    base_batch_dir = os.path.join(BASE_SAVE_DIR, f"batch_{batch_id}")
    os.makedirs(base_batch_dir, exist_ok=True)

    print(f"\n=== BATCH GENERATION: {count} datasets ===")
    for i in range(1, count + 1):
        params = generate_random_inputs(batch_fixed=batch_fixed)
        df = simulate_epidemic(params, use_sir=use_sir)
        run_dir = os.path.join(base_batch_dir, f"run_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)
        save_outputs(df, params, run_dir, save_plots=save_plots)
        print(f"  Dataset {i}/{count} -> {os.path.join(run_dir, 'dataset.csv')}")
    print(f"\nAll datasets saved in: {base_batch_dir}")


def main():
    """
    Main CLI entrypoint for the epidemic dataset generator.
    Parses arguments and dispatches to the appropriate mode.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', nargs='?', default=None)
    parser.add_argument('count', nargs='?', type=int, default=5)
    parser.add_argument('--sir', action='store_true', help='Use SIR model instead of SEIR')
    parser.add_argument('--plots', dest='save_plots', action='store_true', help='Save plots (default)')
    parser.add_argument('--no-plots', dest='save_plots', action='store_false', help='Do not save plots')
    parser.add_argument('--help', action='store_true', help='Show help message and exit')
    # Batch fixed params
    parser.add_argument('--days', type=int, help='Number of days for each dataset in batch')
    parser.add_argument('--mask_score', type=int, help='Mask adherence 1-10')
    parser.add_argument('--multi_wave_count', type=int, help='Number of multi-waves/variants')
    parser.add_argument('--wave_days', type=str, help='Comma-separated days for each wave (e.g. 60,100)')
    parser.add_argument('--wave_betas', type=str, help='Comma-separated beta multipliers for each wave (e.g. 2.5,2.0)')
    parser.add_argument('--wave_seeds', type=str, help='Comma-separated seeds for each wave (e.g. 100,80)')
    parser.set_defaults(save_plots=True)
    args, unknown = parser.parse_known_args()

    if args.help:
        print_help()
        return

    # Mode selection
    if args.mode == 'interact':
        params = get_user_inputs()
        df = simulate_epidemic(params, use_sir=args.sir)
        out_dir = os.path.join(BASE_SAVE_DIR, f"run_{ts()}")
        save_outputs(df, params, out_dir, save_plots=args.save_plots)
    elif args.mode == 'batch':
        count = args.count
        batch_fixed = {
            'days': args.days,
            'mask_score': args.mask_score,
            'multi_wave_count': args.multi_wave_count,
            'wave_days': args.wave_days,
            'wave_betas': args.wave_betas,
            'wave_seeds': args.wave_seeds
        }
        generate_batch(count, use_sir=args.sir, save_plots=args.save_plots, batch_fixed=batch_fixed)
    elif args.mode is None:
        params = generate_random_inputs()
        df = simulate_epidemic(params, use_sir=args.sir)
        out_dir = os.path.join(BASE_SAVE_DIR, f"run_{ts()}")
        save_outputs(df, params, out_dir, save_plots=args.save_plots)
    else:
        print("Unknown mode! Use: 'interact', 'batch <count>', or --help.") 