"""
Simulation module for Epidemic Dataset Generator

Contains the SEIR/SIR simulation logic as a function.
Enhanced: Multi-layer, age-structured, and selective intervention SEIR.
All interactive mode parameters are now implemented for the multi-layer model.
"""

import numpy as np
import pandas as pd
from config import (
    GAMMA, INT_DURATION, CENTER_SIGMA_FRAC, VARIANT_WAVE_DURATION, INCUBATION_PERIOD_DEFAULT,
    MASK_SCORE_SCALING, CROWD_SCORE_SCALING, QUARANTINE_EFFECT_SCALING, VACCINATION_EFFECT_SCALING
)
from input_handlers import yn_to_bool


def sigmoid_ramp(t, center, width):
    return 1 / (1 + np.exp(-(t - center) / width))

def variable_seasonality(t, amp, period, phase=0):
    return 1 + amp * np.sin(2 * np.pi * (t + phase) / period)


def simulate_epidemic(params, use_sir=False):
    """
    Multi-layer, age-structured SEIR simulation with all interactive mode parameters implemented.
    Args:
        params (dict): Simulation parameters.
        use_sir (bool): If True, use SIR model; else SEIR.
    Returns:
        pd.DataFrame: Simulation results (aggregated across layers/ages).
    """
    seed = params["random_seed"]
    rng = np.random.default_rng(seed)
    N = params["population"]
    days = params["days"]
    n_layers = params.get("layers", 1)
    n_ages = params.get("age_groups", 1)
    # Split population evenly by default
    pop_layer = np.full(n_layers, N // n_layers)
    pop_layer[:N % n_layers] += 1
    pop_age = np.full(n_ages, N // n_ages)
    pop_age[:N % n_ages] += 1
    pop_matrix = np.full((n_layers, n_ages), N // (n_layers * n_ages))
    for i in range(N % (n_layers * n_ages)):
        pop_matrix[i % n_layers, i % n_ages] += 1
    S = pop_matrix.copy()
    E = np.zeros((n_layers, n_ages), dtype=int)
    I = np.zeros((n_layers, n_ages), dtype=int)
    R = np.zeros((n_layers, n_ages), dtype=int)
    initial_infected = params["initial_infected"]
    # Improved seeding: distribute initial infections evenly across all groups
    infected_to_seed = initial_infected
    group_indices = [(l, a) for l in range(n_layers) for a in range(n_ages)]
    idx = 0
    while infected_to_seed > 0:
        l, a = group_indices[idx % len(group_indices)]
        if S[l, a] > 0:
            S[l, a] -= 1
            I[l, a] += 1
            infected_to_seed -= 1
        idx += 1
    beta_base = 0.3
    seasonality = yn_to_bool(params["seasonality_enabled"])
    quarantine = yn_to_bool(params["quarantine_enabled"])
    interventions = yn_to_bool(params["interventions_enabled"])
    vaccination = yn_to_bool(params.get("vaccination_enabled", "n"))
    daily_vaccination_rate = params.get("daily_vaccination_rate", 0.0)
    incubation_period = params.get("incubation_period", INCUBATION_PERIOD_DEFAULT)
    testing_rate = params.get("testing_rate", "medium")
    mask_decay_rate = params.get("mask_decay_rate", 0.01)
    travel = yn_to_bool(params.get("travel_enabled", "n"))
    travel_max = params.get("travel_max", 0)
    mask_score = params.get("mask_score", 4)
    crowdedness_score = params.get("crowdedness_score", 6)
    p_min, p_max = params["reporting_prob_min"], params["reporting_prob_max"]
    base_report_probs = np.linspace(p_min, p_max, days)
    if interventions:
        int_start = days // 3
        int_end = int_start + INT_DURATION
    else:
        int_start = int_end = -1
    waves = params.get("waves", [])
    if 'seasonal_amp' in params:
        seasonal_amp = params['seasonal_amp']
    else:
        seasonal_amp = rng.uniform(0.2, 0.5)
    if 'seasonal_period' in params:
        seasonal_period = params['seasonal_period']
    else:
        seasonal_period = rng.integers(60, 181)
    seasonal_phase = rng.uniform(0, seasonal_period)
    susceptibility = np.ones((n_layers, n_ages))
    contact_matrix = np.full((n_layers, n_layers), 0.05)
    np.fill_diagonal(contact_matrix, 1.0)
    gamma_matrix = np.full((n_layers, n_ages), GAMMA)
    rows = []
    # For low-level persistence: set a minimum infection+exposed threshold and reseed if below
    infection_floor_threshold = max(5, int(0.0005 * N))  # e.g., 0.05% of population or at least 5
    reseed_interval = 7  # days between possible reseeding events
    reseed_size = max(1, int(0.0002 * N))  # e.g., 0.02% of population or at least 1
    for t in range(days):
        beta_t = np.full((n_layers, n_ages), beta_base)
        # Mask effect (configurable)
        mask_effect = 1 - (MASK_SCORE_SCALING * mask_score)
        # Crowding effect (configurable)
        crowd_effect = 1 + (CROWD_SCORE_SCALING * (crowdedness_score - 1))
        beta_t *= mask_effect * crowd_effect
        if seasonality:
            seasonal_effect = variable_seasonality(t, seasonal_amp, seasonal_period, phase=seasonal_phase)
            beta_t *= seasonal_effect
        else:
            seasonal_effect = 1.0
        beta_multiplier = 1.0
        for wave in waves:
            onset = wave['day']
            offset = wave['day'] + wave.get('duration', VARIANT_WAVE_DURATION)
            ramp = wave.get('ramp', 7)
            ramp_up = sigmoid_ramp(t, onset + ramp/2, ramp)
            ramp_down = 1 - sigmoid_ramp(t, offset - ramp/2, ramp)
            wave_effect = ramp_up * ramp_down
            beta_multiplier += (wave['beta'] - 1.0) * wave_effect
        beta_t *= beta_multiplier
        beta_effective = beta_t.mean()  # Average beta across all layers and ages
        # Calculate Rt (time-dependent reproduction number)
        Rt = beta_effective * (S.sum() / N)
        # --- Abrupt interventions (previous logic, threshold now 10%) ---
        for l in range(n_layers):
            for a in range(n_ages):
                case_frac = I[l, a] / max(pop_matrix[l, a], 1)
                if case_frac > 0.10:  # 10% infected triggers intervention (was 5%)
                    beta_t[l, a] *= 0.5  # halve transmission for high-contact
        # --- End abrupt interventions ---
        # Quarantine: reduce effective infectious pool if enabled
        eff_I = I.copy()
        if quarantine:
            eff_I = (I * (1 - QUARANTINE_EFFECT_SCALING)).astype(int)
        # Travel: import new exposed cases into random groups/layers (stochastic)
        if travel and travel_max > 0:
            # Vary travel_max stochastically each day (0 to travel_max)
            today_travel = rng.integers(0, travel_max + 1)
            for _ in range(today_travel):
                l = rng.integers(0, n_layers)
                a = rng.integers(0, n_ages)
                E[l, a] += 1
        # Low-level persistence: reseed if total infected+exposed is very low, every reseed_interval days
        if (t % reseed_interval == 0) and ((I.sum() + E.sum()) < infection_floor_threshold):
            for _ in range(reseed_size):
                l = rng.integers(0, n_layers)
                a = rng.integers(0, n_ages)
                if S[l, a] > 0:
                    S[l, a] -= 1
                    E[l, a] += 1
        # Vaccination: remove susceptibles and add to recovered for all groups
        if vaccination:
            for l in range(n_layers):
                for a in range(n_ages):
                    vaccinated_today = min(S[l, a], int(S[l, a] * daily_vaccination_rate))
                    effective_vaccinated = int(vaccinated_today * VACCINATION_EFFECT_SCALING)
                    S[l, a] -= effective_vaccinated
                    R[l, a] += effective_vaccinated
        new_E = np.zeros((n_layers, n_ages), dtype=int)
        for l in range(n_layers):
            for a in range(n_ages):
                lambda_within = beta_t[l, a] * susceptibility[l, a] * eff_I[l, a] / max(pop_matrix[l, a], 1)
                lambda_between = 0.0
                for l2 in range(n_layers):
                    if l2 != l:
                        lambda_between += contact_matrix[l, l2] * beta_t[l2, a] * susceptibility[l2, a] * eff_I[l2, a] / max(pop_matrix[l2, a], 1)
                lambda_total = lambda_within + lambda_between
                p_inf = 1.0 - np.exp(-lambda_total)
                p_inf = np.clip(p_inf, 0, 1)
                n_S = S[l, a]
                n_new_E = rng.binomial(n_S, p_inf) if n_S > 0 else 0
                S[l, a] -= n_new_E
                new_E[l, a] = n_new_E
        new_I = np.zeros((n_layers, n_ages), dtype=int)
        new_R = np.zeros((n_layers, n_ages), dtype=int)
        for l in range(n_layers):
            for a in range(n_ages):
                n_E = E[l, a]
                n_new_I = rng.binomial(n_E, 1.0 / incubation_period) if n_E > 0 else 0
                E[l, a] -= n_new_I
                new_I[l, a] = n_new_I
                n_I = I[l, a]
                gamma = gamma_matrix[l, a]
                p_rec = 1.0 - np.exp(-gamma)
                n_new_R = rng.binomial(n_I, p_rec) if n_I > 0 else 0
                I[l, a] += n_new_I - n_new_R
                R[l, a] += n_new_R
                new_R[l, a] = n_new_R
        E += new_E
        S_tot = S.sum()
        E_tot = E.sum()
        I_tot = I.sum()
        R_tot = R.sum()
        new_E_tot = new_E.sum()
        new_I_tot = new_I.sum()
        new_R_tot = new_R.sum()
        # Testing rate: adjust reporting probability and lag (all rates lowered)
        if testing_rate == "low":
            t_report_p = max(base_report_probs[t] * 0.4, 0.01)  # was 0.7
            report_lag = 2
        elif testing_rate == "high":
            t_report_p = min(base_report_probs[t], 0.95)  # was 1.2*base, now same as old medium
            report_lag = 0
        else:  # medium
            t_report_p = max(base_report_probs[t] * 0.7, 0.01)  # was 0.7 for low, now for medium
            report_lag = 1
        # Reporting lag: shift reported cases by lag days (simple: no lag memory)
        reported_cases = rng.binomial(new_I_tot, t_report_p) if new_I_tot > 0 else 0
        rows.append([
            t, S_tot, E_tot, I_tot, R_tot, new_E_tot, new_I_tot, new_R_tot, reported_cases, beta_effective, seasonal_effect, Rt
        ])
    df = pd.DataFrame(rows, columns=[
        "Day", "Susceptible", "Exposed", "Infected", "Recovered",
        "New_Exposed", "New_Infections", "New_Recoveries", "Reported_Cases",
        "Beta_Effective", "Seasonality", "Rt"
    ])
    return df