"""
Plotting module for Epidemic Dataset Generator

Contains functions to plot SEIR compartments and reported cases.
"""

import matplotlib.pyplot as plt
import os
from config import BASE_SAVE_DIR


def ensure_dir(path: str):
    """
    Ensure that a directory exists; create if not.
    """
    os.makedirs(path, exist_ok=True)


def plot_sir(df, path):
    """
    Plot SEIR compartments as a stacked area plot and save to file.
    Args:
        df (pd.DataFrame): Simulation results.
        path (str): Output file path for the plot.
    """
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(10, 6))
    plt.stackplot(df["Day"],  df["Infected"], df["Exposed"], df["Susceptible"],  df["Recovered"],
                  labels=["Infected","Exposed", "Susceptible",  "Recovered"],
                  colors=["#ff6961",  "#fdfd96","#77b5fe", "#77dd77"])
    plt.legend(loc="upper right")
    plt.title("Epidemic Simulation (SEIR Compartments)")
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reported(df, path):
    """
    Plot reported cases over time and save to file.
    Args:
        df (pd.DataFrame): Simulation results.
        path (str): Output file path for the plot.
    """
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(10, 5))
    plt.plot(df["Day"], df["Reported_Cases"], label="Reported Cases", color="orange")
    plt.title("Reported Cases Over Time")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close() 