"""
Output module for Epidemic Dataset Generator

Handles saving simulation results (CSV, JSON) and plots.
"""

import os
import json
from config import BASE_SAVE_DIR
from plotting import plot_sir, plot_reported, ensure_dir


def save_outputs(df, params, out_dir, save_plots=True):
    """
    Save simulation results to CSV, JSON, and optionally plots.
    Args:
        df (pd.DataFrame): Simulation results.
        params (dict): Simulation parameters.
        out_dir (str): Output directory.
        save_plots (bool): Whether to save plots.
    """
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "dataset.csv")
    json_path = os.path.join(out_dir, "params.json")
    sir_plot = os.path.join(out_dir, "stacked_sir.png")
    reported_plot = os.path.join(out_dir, "reported_cases.png")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(params, f, indent=4)

    if save_plots:
        plot_sir(df, sir_plot)
        plot_reported(df, reported_plot)
        print(f"  Plots: {sir_plot}, {reported_plot}")

    print(f"\nData and plots saved in: {out_dir}")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}") 