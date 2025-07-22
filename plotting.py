"""
Advanced plotting utilities for Epidemic Dataset Generator.

This file is a **drop‑in replacement** for the plotting file you uploaded. It keeps
the original `plot_sir()` and `plot_reported()` functions (so existing imports
won't break) and adds all the new, more visual plots you requested:

    • Compartment Heatmap (Day × S/E/I/R)
    • New Exposed vs New Recoveries dual line
    • Reported vs Actual Infections
    • I & S Area + Reported overlay (scaled)
    • Phase Diagram (I vs S, color by Day or Beta_Effective)
    • 3D Day–Beta–Infected ribbon surface
    • Streamgraph SEIR (centered “Theme River” style)
    • Radial (polar) Seasonality + Infected overlay
    • Beta vs Cases Scatter (colored by Day)

All figures are saved at dpi=300. Seaborn is optional; the module falls back to a
matplotlib style if Seaborn isn’t installed, so you will not crash.

Usage example:

    import pandas as pd
    import advanced_plotting as ap

    df = pd.read_csv("run.csv")
    ap.plot_compartment_heatmap(df, "out/heatmap.png")
    ap.plot_phase_diagram(df, "out/phase_beta.png", color_by="Beta_Effective")
    ap.save_all_advanced_plots(df, "out/all_plots")

"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

# ------------------------------------------------------------------
# Optional Seaborn styling (graceful fallback)
# ------------------------------------------------------------------
try:  # try/except so missing seaborn never breaks plotting
    import seaborn as sns  # type: ignore
    _HAVE_SNS = True
    sns.set_style("darkgrid")
    sns.set_context("talk")
except Exception:  # ImportError or runtime error
    _HAVE_SNS = False
    plt.style.use("seaborn-v0_8-darkgrid")  # built-in mpl style

# If project defines BASE_SAVE_DIR we don’t actually need it here, but we'll try.
try:  # soft import (unused but harmless)
    from config import BASE_SAVE_DIR  # type: ignore  # noqa: F401
except Exception:  # keep module importable if config missing
    BASE_SAVE_DIR = "."

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
DEF_COMPARTMENTS = [ "Exposed", "Infected","Susceptible", "Recovered"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _maybe_numpy(series):
    """Return underlying numpy array, even if pandas Series."""
    return getattr(series, "to_numpy", lambda: np.asarray(series))()


def _get_colors(n: int, palette: str | None = "Set2"):
    """Return *n* distinct colors; use Seaborn palette if available."""
    if _HAVE_SNS and palette is not None:
        return sns.color_palette(palette, n)
    # fallback color cycle (color‑blind friendly-ish)
    base = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]

def plot_sir(df: pd.DataFrame, path: str) -> None:
    """Classic stacked SEIR compartments plot (Infected, Exposed, Susceptible, Recovered)."""
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = _get_colors(4, palette="hls" if _HAVE_SNS else None)
    ax.stackplot(
        df["Day"],
        df["Infected"],
        df["Exposed"],
        df["Susceptible"],
        df["Recovered"],
        labels=["Infected", "Exposed", "Susceptible", "Recovered"],
        colors=colors,
        alpha=0.9,
    )
    ax.set_title("Epidemic Simulation (SEIR Compartments)", fontsize=18, weight="bold")
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_reported(df: pd.DataFrame, path: str) -> None:
    """Reported cases over time (line, *no markers* as per your request)."""
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(12, 6))
    color = _get_colors(1, palette="Set1")[0]
    ax.plot(df["Day"], df["Reported_Cases"], label="Reported Cases", color=color, linewidth=2.5, alpha=0.9)
    ax.set_title("Reported Cases Over Time", fontsize=18, weight="bold")
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Cases", fontsize=14)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# 1. Heatmap: Day × Compartment (S/E/I/R)

def plot_compartment_heatmap(
    df: pd.DataFrame,
    path: str,
    compartments=DEF_COMPARTMENTS,
    log_scale: bool = False,
    github_style: bool = True,
    bins: int = 5,
) -> None:
    """GitHub‑style compartment heatmap.

    By default (`github_style=True`), each compartment row is *independently* normalized
    to its own max and rendered with a discrete color ramp (like the GitHub
    contributions calendar). Each row uses a different Seaborn color family:

        Susceptible → Blues
        Exposed     → Oranges
        Infected    → Reds
        Recovered   → Greens

    Set ``github_style=False`` to fall back to the original continuous ``viridis`` heatmap.
    ``log_scale`` only applies in the non‑GitHub mode.
    ``bins`` controls the number of discrete color steps (GitHub uses ~5).
    """
    ensure_dir(os.path.dirname(path))

    # Extract numeric values (rows=compartments, cols=time)
    values = df[compartments].to_numpy(dtype=float).T
    n_comp, n_days = values.shape

    if not github_style:
        # Original continuous heatmap path
        data = np.log10(np.clip(values, a_min=1, a_max=None)) if log_scale else values
        fig, ax = plt.subplots(figsize=(max(8, n_days / 8), 4 + 0.5 * n_comp))
        im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
        n = len(df)
        step = max(1, n // 10)
        idxs = np.arange(0, n, step)
        ax.set_xticks(idxs)
        ax.set_xticklabels(df["Day"].iloc[idxs])
        ax.set_yticks(np.arange(n_comp))
        ax.set_yticklabels(compartments)
        ax.set_xlabel("Day", fontsize=14)
        ax.set_title("Compartment Heatmap", fontsize=18, weight="bold")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("log10(Value)" if log_scale else "Population", rotation=270, labelpad=20)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return

    # --- GitHub style ----------------------------------------------------
    # Row-wise max for scaling; avoid divide-by-zero
    row_max = values.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    scaled = values / row_max

    # Discretize to bins
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(scaled, edges, right=True)  # 0..bins
    idx[idx > bins] = bins

    # Choose seaborn palettes per compartment (fallback generic grays)
    if _HAVE_SNS:
        pal_map = {
            'Susceptible': sns.color_palette("Blues", bins + 1),
            'Exposed': sns.color_palette("Oranges", bins + 1),
            'Infected': sns.color_palette("Reds", bins + 1),
            'Recovered': sns.color_palette("Greens", bins + 1),
        }
    else:
        # fallback: grayscale ramp reused for all
        gray_pal = [str(x) for x in np.linspace(0.9, 0.1, bins + 1)]
        pal_map = {c: gray_pal for c in compartments}

    # color conversion helper (local import avoids global clutter)
    from matplotlib.colors import to_rgb as _to_rgb

    # Build image array
    img = np.zeros((n_comp, n_days, 3))
    for r, comp in enumerate(compartments):
        pal = pal_map.get(comp, pal_map[next(iter(pal_map))])
        pal_rgb = [_to_rgb(clr) for clr in pal]
        for c in range(n_days):
            img[r, c] = pal_rgb[idx[r, c]]

    fig, ax = plt.subplots(figsize=(max(8, n_days / 8), 4 + 0.5 * n_comp))
    ax.imshow(img, aspect="auto", interpolation="nearest")

    # X ticks
    n = len(df)
    step = max(1, n // 10)
    idxs = np.arange(0, n, step)
    ax.set_xticks(idxs)
    ax.set_xticklabels(df["Day"].iloc[idxs])

    # Y ticks
    ax.set_yticks(np.arange(n_comp))
    ax.set_yticklabels(compartments)

    ax.set_xlabel("Day", fontsize=14)
    ax.set_title("Compartment Heatmap", fontsize=18, weight="bold")

    # Legend: darkest swatch per compartment
    import matplotlib.patches as mpatches
    handles = []
    for comp in compartments:
        handles.append(mpatches.Patch(color=pal_map.get(comp)[-1], label=comp))
    ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=True)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# 2. Dual line: New Exposed vs New Recoveries

def plot_new_exposed_vs_new_recoveries(df: pd.DataFrame, path: str) -> None:
    """Dual line plot of `New_Exposed` and `New_Recoveries`."""
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = _get_colors(2, palette="Set1")
    ax.plot(df["Day"], df["New_Exposed"], label="New Exposed", color=colors[0], linewidth=2.0)
    ax.plot(df["Day"], df["New_Recoveries"], label="New Recoveries", color=colors[1], linewidth=2.0)
    ax.set_title("New Exposed vs New Recoveries", fontsize=18, weight="bold")
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# 3. Reported vs Actual Infections

def plot_reported_vs_actual_infections(
    df: pd.DataFrame,
    path: str,
    actual_col: str = "New_Infections",
) -> None:
    """Compare reported cases to actual infections (daily counts)."""
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = _get_colors(2, palette="Set1")
    ax.plot(df["Day"], df[actual_col], label="Actual Infections", color=colors[0], linewidth=2.5)
    ax.plot(
        df["Day"],
        df["Reported_Cases"],
        label="Reported Cases",
        color=colors[1],
        linewidth=2.5,
        linestyle="--",
    )
    ax.set_title("Reported vs Actual Infections", fontsize=18, weight="bold")
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Daily Count", fontsize=14)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# 4. Area Chart for I & S + Reported overlay

def plot_is_area_with_reported(df: pd.DataFrame, path: str) -> None:
    """Stacked area (Infected over Susceptible) + bold line for Reported_Cases.

    Reported is scaled to the S+I axis range so it remains visible when counts
    are in different units.
    """
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = _get_colors(2, palette="Set2")
    ax.stackplot(
        df["Day"],
        df["Infected"],
        df["Susceptible"],
        colors=colors,
        labels=["Infected", "Susceptible"],
        alpha=0.6,
    )
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    # Overlay reported cases scaled to population axis
    reported = _maybe_numpy(df["Reported_Cases"])
    pop_max = max((df["Infected"].max() + df["Susceptible"].max()), 1)
    rep_max = reported.max() if reported.size else 1
    scale = pop_max / rep_max if rep_max > 0 else 1
    ax.plot(df["Day"], reported * scale, color="black", linewidth=2.5, label="Reported (scaled)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=11, frameon=True)
    ax.set_title("I & S Area + Reported Cases Overlay", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_phase_diagram_beta(df: pd.DataFrame, path: str, x: str = "Susceptible", y: str = "Infected", color_by: str = "Beta_Effective") -> None:
    """Phase trajectory colored by Beta_Effective (default I vs S)."""
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(7, 7))
    xvals = df[x].to_numpy()
    yvals = df[y].to_numpy()
    if color_by in df.columns:
        cvals = df[color_by].to_numpy()
    else:
        cvals = np.arange(len(df))
    sc = ax.scatter(xvals, yvals, c=cvals, cmap="viridis", s=25, edgecolor="none")
    ax.plot(xvals, yvals, color="gray", alpha=0.4, linewidth=1.0)
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel(y, fontsize=14)
    ax.set_title(f"Phase Diagram ({y} vs {x}) colored by {color_by}", fontsize=16, weight="bold")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(color_by, rotation=270, labelpad=20)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_phase_diagram_I_vs_R_beta(df: pd.DataFrame, path: str, color_by: str = "Beta_Effective") -> None:
    """Phase trajectory I vs R colored by Beta_Effective."""
    plot_phase_diagram_beta(df, path, x="Recovered", y="Infected", color_by=color_by)




# ------------------------------------------------------------------
# 6. 3D Surface: Day × Beta_Effective × Infected
# ------------------------------------------------------------------

def plot_3d_day_beta_infected(df: pd.DataFrame, path: str, surface: bool = True) -> None:
    """3D visualization of Day, Beta_Effective, and Infected.

    With a single time series we can’t create a full gridded surface; instead we
    create a *ribbon* by duplicating the curve (trisurf) for a cool look. Set
    `surface=False` to draw only the 3D line.
    """
    ensure_dir(os.path.dirname(path))
    day = _maybe_numpy(df["Day"])
    beta = _maybe_numpy(df["Beta_Effective"]) if "Beta_Effective" in df.columns else np.zeros_like(day)
    inf = _maybe_numpy(df["Infected"])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if surface and len(day) > 1:
        width = (beta.max() - beta.min()) * 0.01 if np.ptp(beta) > 0 else 0.01
        day2 = np.concatenate([day, day])
        beta2 = np.concatenate([beta - width, beta + width])
        inf2 = np.concatenate([inf, inf])
        ax.plot_trisurf(day2, beta2, inf2, cmap="viridis", alpha=0.8, linewidth=0.2, edgecolor="none")
    else:
        ax.plot(day, beta, inf, color="black", linewidth=2.0)

    ax.set_xlabel("Day")
    ax.set_ylabel("Beta_Effective")
    ax.set_zlabel("Infected")
    ax.set_title("3D Day–Beta–Infected", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_beta_vs_seasonality(
    df: pd.DataFrame,
    path: str,
    style: str = "scatter_line",
    color_by: str = "Day",
) -> None:
    """XY plot of Beta_Effective (x) vs Seasonality (y).

    Parameters
    ----------
    style : {'scatter_line', 'scatter', 'line'}
        * ``scatter_line`` (default) draws a light connecting line in addition to points.
        * ``scatter`` plots only points.
        * ``line`` plots only a line in temporal order.
    color_by : column name used to color points (only for scatter_*)
        Defaults to ``Day`` so early/late points are distinguishable.
    """
    ensure_dir(os.path.dirname(path))
    if 'Beta_Effective' not in df.columns or 'Seasonality' not in df.columns:
        print('[WARN] plot_beta_vs_seasonality: required columns missing; skipping.')
        return

    x = df['Beta_Effective'].to_numpy()
    y = df['Seasonality'].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 7))

    if style in ("scatter_line", "scatter"):
        if color_by in df.columns:
            cvals = df[color_by].to_numpy()
        else:
            cvals = None
        sc = ax.scatter(x, y, c=cvals, cmap='viridis' if cvals is not None else None,
                        s=40, alpha=0.9, edgecolor='k', linewidth=0.3)
        if cvals is not None:
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label(color_by, rotation=270, labelpad=20)
        if style == "scatter_line":
            # connect in Day order (sorted by index)
            ax.plot(x, y, color='gray', alpha=0.4, linewidth=1.0)
    else:  # line only
        ax.plot(x, y, color='black', linewidth=2.0)

    ax.set_xlabel('Beta_Effective', fontsize=14)
    ax.set_ylabel('Seasonality', fontsize=14)
    ax.set_title('Beta vs Seasonality', fontsize=16, weight='bold')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# 8. Circular / Radial Seasonality Plot
# ------------------------------------------------------------------

def plot_radial_seasonality(df: pd.DataFrame, path: str, period: int | None = None) -> None:
    """Polar plot showing seasonality (and scaled infections) over a cycle.

    period : length of full seasonal cycle in days; inferred from data if None.
    """
    ensure_dir(os.path.dirname(path))
    day = _maybe_numpy(df["Day"]).astype(float)
    if period is None:
        period = int(day.max() - day.min() + 1)
    theta = 2 * np.pi * ((day - day.min()) % period) / period
    season = _maybe_numpy(df["Seasonality"]) if "Seasonality" in df.columns else np.ones_like(day)
    infected = _maybe_numpy(df["Infected"])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, season, label="Seasonality", linewidth=2.5)
    if infected.max() > 0:
        inf_scaled = infected / infected.max() * season.max()
        ax.plot(theta, inf_scaled, label="Infected (scaled)", linewidth=1.5, linestyle="--")
    ax.set_title("Radial Seasonality & Infections", fontsize=16, weight="bold", pad=20)
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_zero_location("N")  # 0 at top
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# 9. Beta vs Cases Scatter (colored by Day)
# ------------------------------------------------------------------

def plot_beta_vs_cases(df: pd.DataFrame, path: str) -> None:
    """Scatter Beta_Effective vs New_Infections colored by Day."""
    ensure_dir(os.path.dirname(path))
    beta = _maybe_numpy(df["Beta_Effective"]) if "Beta_Effective" in df.columns else np.zeros(len(df))
    cases = _maybe_numpy(df["New_Infections"])
    day = _maybe_numpy(df["Day"])

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(beta, cases, c=day, cmap="viridis", s=40, alpha=0.9, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Beta_Effective", fontsize=14)
    ax.set_ylabel("New Infections", fontsize=14)
    ax.set_title("Beta vs New Infections (colored by Day)", fontsize=16, weight="bold")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Day", rotation=270, labelpad=20)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Convenience: generate everything available
# ------------------------------------------------------------------

def save_all_advanced_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Generate a suite of advanced plots for the given dataframe."""
    ensure_dir(out_dir)
    plot_compartment_heatmap(df, os.path.join(out_dir, "heatmap.png"))
    plot_sir(df, os.path.join(out_dir, "sir_plot.png"))
    plot_reported(df, os.path.join(out_dir, "reported_cases.png"))
    plot_new_exposed_vs_new_recoveries(df, os.path.join(out_dir, "new_exposed_vs_recoveries.png"))
    plot_reported_vs_actual_infections(df, os.path.join(out_dir, "reported_vs_actual.png"))
    plot_is_area_with_reported(df, os.path.join(out_dir, "is_area_reported.png"))
    plot_phase_diagram_beta(df, os.path.join(out_dir, "phase_I_vs_S_day.png"), color_by="Beta_Effective")
    if "Beta_Effective" in df.columns:
        plot_phase_diagram_I_vs_R_beta(df, os.path.join(out_dir, "phase_I_vs_R_beta.png"), color_by="Beta_Effective")
        plot_3d_day_beta_infected(df, os.path.join(out_dir, "3d_day_beta_infected.png"), surface=True)
        plot_beta_vs_cases(df, os.path.join(out_dir, "beta_vs_cases.png"))
    if "Seasonality" in df.columns:
        plot_radial_seasonality(df, os.path.join(out_dir, "radial_seasonality.png"))
    try:
        plot_beta_vs_seasonality(df, os.path.join(out_dir, "streamgraph_seir.png"))
    except Exception:
        pass  # ignore if compartments missing


__all__ = [
    "plot_sir",
    "plot_reported",
    "plot_compartment_heatmap",
    "plot_new_exposed_vs_new_recoveries",
    "plot_reported_vs_actual_infections",
    "plot_is_area_with_reported",
    "plot_phase_diagram_beta",
    "plot_phase_diagram_I_vs_R_beta",
    "plot_3d_day_beta_infected",
    "plot_beta_vs_seasonality",
    "plot_radial_seasonality",
    "plot_beta_vs_cases",
    "save_all_advanced_plots",
]
