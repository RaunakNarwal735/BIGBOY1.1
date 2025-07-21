"""
Simulation module for Epidemic Dataset Generator

Contains the SEIR/SIR simulation logic as a function.
"""

import numpy as np
import pandas as pd
from config import (
    GAMMA, SEASONAL_PERIOD, INT_DURATION, MULTIWAVE_DURATION, CENTER_SIGMA_FRAC,
    VARIANT_WAVE_DURATION, INCUBATION_PERIOD_DEFAULT
)
from input_handlers import yn_to_bool


def simulate_epidemic(params, use_sir=False):
    """
    Run stochastic epidemic simulation using modified SIR/SEIR dynamics.

    Args:
        params (dict): Simulation parameters.
        use_sir (bool): If True, use SIR model; else SEIR.

    Returns:
        pd.DataFrame: Simulation results with agreed column schema.
    """
    # RNG
    seed = params["random_seed"]
    rng = np.random.default_rng(seed)

    # Unpack
    N = params["population"]
    days = params["days"]
    S = N - params["initial_infected"]
    I = params["initial_infected"]
    R = 0

    # Base beta
    beta_base = 0.3

    # Flags
    seasonality = yn_to_bool(params["seasonality_enabled"])
    quarantine = yn_to_bool(params["quarantine_enabled"])
    interventions = yn_to_bool(params["interventions_enabled"])
    multi_wave = yn_to_bool(params["multi_wave"])

    # Reporting schedule + jitter
    p_min, p_max = params["reporting_prob_min"], params["reporting_prob_max"]
    base_report_probs = np.linspace(p_min, p_max, days)

    # Intervention window(auto)
    if interventions:
        int_start = days // 3
        int_end = int_start + INT_DURATION
    else:
        int_start = int_end = -1  # never triggers

    # Multi-wave bump
    if multi_wave:
        wave_start = days // 2
        wave_end = wave_start + MULTIWAVE_DURATION
    else:
        wave_start = wave_end = -1

    # Central Gaussian envelope to push peak toward middle
    center_mu = days / 2
    center_sigma = days * CENTER_SIGMA_FRAC  # convert fraction to days

    # Random dropper scheduler
    # (handled in main loop)

    # Unpack new params
    vaccination = yn_to_bool(params.get("vaccination_enabled", "n"))
    daily_vaccination_rate = params.get("daily_vaccination_rate", 0.0)
    incubation_period = params.get("incubation_period", INCUBATION_PERIOD_DEFAULT)
    testing_rate = params.get("testing_rate", "medium")
    mask_decay_rate = params.get("mask_decay_rate", 0.01)
    travel = yn_to_bool(params.get("travel_enabled", "n"))
    travel_max = params.get("travel_max", 0)
    waves = params.get("waves", [])
    mask_score = params.get("mask_score", 4)
    crowdedness_score = params.get("crowdedness_score", 6)

    # SEIR compartments
    E = 0
    I = params["initial_infected"]
    R = 0
    E_queue = [0] * incubation_period  # queue for exposed individuals

    # Storage
    rows = []
    beta_rw = 0.0  # random walk component for beta
    gamma_rw = 0.0  # random walk component for gamma
    for t in range(days):
        # --- Realistic beta build-up ---
        # Mask effect (0–50% reduction)
        mask_effect = 1 - 0.05 * mask_score
        # Crowding effect (0–45% increase)
        crowd_effect = 1 + 0.05 * (crowdedness_score - 1)
        # Base beta
        beta_t = beta_base * mask_effect * crowd_effect
        # Seasonality
        if seasonality:
            beta_t *= 1 + 0.2 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
        # Intervention
        if int_start <= t <= int_end:
            beta_t *= 0.5
        # Multi-wave/variant
        beta_multiplier = 1.0
        for wave in waves:
            if wave["day"] <= t < wave["day"] + VARIANT_WAVE_DURATION:
                beta_multiplier *= wave["beta"]
        beta_t *= beta_multiplier
        # Daily jitter (10%)
        beta_t *= (1 + rng.normal(0, 0.10))
        # Add slow random walk to beta (stronger)
        beta_rw += rng.normal(0, 0.02)
        beta_rw = np.clip(beta_rw, -0.3, 0.3)
        beta_t *= (1 + beta_rw)

        # Quarantine effect (50% reduction)
        if quarantine:
            eff_I = I * 0.5
            q_frac = 0.5
        else:
            eff_I = I
            q_frac = 0.0

        # Minimum infection floor: if I is very low and S is available, seed a few new infections
        if I < 5 and S > 10:
            new_seed = min(20, S)
            S -= new_seed
            E += new_seed

        # Vaccination (90% efficacy)
        if vaccination and S > 0:
            vaccinated_today = min(S, int(S * daily_vaccination_rate))
            effective_vaccinated = int(vaccinated_today * 0.9)
            S -= effective_vaccinated
            R += effective_vaccinated
            # 10% remain susceptible (breakthroughs)

        # Travel importation: add travel_max new exposed from outside
        if travel and travel_max > 0:
            new_travelers = travel_max
            E += new_travelers
            E_queue[-1] += new_travelers

        # Inject new exposed for each wave on its start day
        for wave in waves:
            if t == wave["day"] and S > 0:
                new_wave_cases = min(wave["seed"], S)
                S -= new_wave_cases
                E += new_wave_cases
                E_queue[-1] += new_wave_cases

        # Apply all active waves (variants): boost beta for duration
        beta_multiplier = 1.0
        for wave in waves:
            if wave["day"] <= t < wave["day"] + VARIANT_WAVE_DURATION:
                beta_multiplier *= wave["beta"]
        beta_t *= beta_multiplier

        # Transmission & recovery probabilities
        p_inf = 1.0 - np.exp(-beta_t * eff_I / N)
        p_inf = np.clip(p_inf, 0, 1)

        # Gamma (recovery rate) with random walk and daily jitter (stronger)
        gamma_t = GAMMA * (1 + gamma_rw + rng.normal(0, 0.10))  # 10% daily jitter
        gamma_rw += rng.normal(0, 0.01)
        gamma_rw = np.clip(gamma_rw, -0.15, 0.15)
        p_rec = 1.0 - np.exp(-gamma_t)
        p_rec = np.clip(p_rec, 0, 1)

        # Robust minimum infected floor: every 3 days, if both I and E are zero and S > 10, seed 15–30 new infections (split E/I)
        if t > 0 and t % 3 == 0 and I + E == 0 and S > 10:
            new_seed = min(rng.integers(15, 31), S)
            S -= new_seed
            e_seed = new_seed // 2
            i_seed = new_seed - e_seed
            E += e_seed
            I += i_seed
            E_queue[-1] += e_seed

        # New transitions (stochastic)
        new_exp = rng.binomial(S, p_inf) if S > 0 else 0
        S -= new_exp
        E_queue.append(new_exp)
        new_inf = E_queue.pop(0)
        # E->I floor: always if E > 0
        if new_inf == 0 and E > 0:
            new_inf = 1
            E -= 1
        E += new_exp - new_inf
        new_rec = rng.binomial(I, p_rec) if I > 0 else 0
        I += new_inf - new_rec
        R += new_rec

        # Random dropper: at random intervals, add random number of new exposed
        # (handled externally in main loop if needed)

        # Testing intensity affects reporting probability and lag
        if testing_rate == "low":
            report_p = max(base_report_probs[t] * 0.7, 0.01)
            report_lag = 2
        elif testing_rate == "high":
            report_p = min(base_report_probs[t] * 1.2, 0.99)
            report_lag = 0
        else:
            report_p = base_report_probs[t]
            report_lag = 1
        # Reporting lag: shift reported cases by lag days
        if t >= report_lag:
            reported_cases = rng.binomial(new_inf, report_p) if new_inf > 0 else 0
        else:
            reported_cases = 0

        # Set output flags for DataFrame
        season_index = 1.0
        if seasonality:
            season_index = 1 + 0.2 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
        int_flag = 1 if int_start <= t <= int_end else 0
        wave_flag = 0
        for wave in waves:
            if wave["day"] <= t < wave["day"] + VARIANT_WAVE_DURATION:
                wave_flag = 1
                break

        # Effective reproduction number Rt
        Rt = (beta_t / GAMMA) * (S / N) if N > 0 else 0

        # Store
        rows.append([
            t, S, E, I, R, new_exp, new_inf, new_rec, reported_cases,
            beta_t, season_index, int_flag, q_frac, report_p, wave_flag, Rt
        ])

    df = pd.DataFrame(rows, columns=[
        "Day", "Susceptible", "Exposed", "Infected", "Recovered",
        "New_Exposed", "New_Infections", "New_Recoveries", "Reported_Cases",
        "Beta_Effective", "Season_Index", "Intervention_Flag",
        "Quarantine_Fraction", "Reporting_Prob", "MultiWave_Flag", "Rt"
    ])
    return df 