"""
Input Handlers for Epidemic Dataset Generator

Contains functions for interactive user input and random parameter generation.
"""

import random
from datetime import datetime
from config import (
    INCUBATION_PERIOD_DEFAULT, MASK_DECAY_RATE_DEFAULT, default_vaccination_rate
)

# =============================
# Helper Functions
# =============================
def yn_to_bool(v: str) -> bool:
    """
    Convert a yes/no string to boolean.
    Accepts: 'y', 'yes', 'true', '1' as True; else False.
    """
    return str(v).strip().lower() in ("y", "yes", "true", "1")


def get_user_inputs():
    """
    Prompt the user for all simulation parameters interactively.
    Returns a dictionary of parameters.
    """
    print("\n=== INTERACTIVE MODE ===")

    # population
    pop = input("Population [73500]: ").strip()
    population = int(pop) if pop else 73500
    if population > 1_000_000:
        print("Population capped at 1,000,000.")
        population = 1_000_000
    if population < 1000:
        print("Population too low; forcing 1000.")
        population = 1000

    # days
    d = input("Number of days [180]: ").strip()
    days = int(d) if d else 180
    if days < 30:
        print("Days too low; forcing 30.")
        days = 30

    # initial infected
    ii = input("Initial infected [50]: ").strip()
    initial_infected = int(ii) if ii else 50
    max_init = max(1, min(population // 20, 5000))  # <=5% of pop, cap 5000
    if initial_infected > max_init:
        print(f"Initial infected capped at {max_init}.")
        initial_infected = max_init
    if initial_infected < 1:
        initial_infected = 1

    # mask and crowd
    ms = input("Mask adherence 1-10 [4]: ").strip()
    mask_score = int(ms) if ms else 4
    mask_score = min(max(mask_score, 1), 10)

    cs = input("Crowdedness 1-10 [6]: ").strip()
    crowdedness_score = int(cs) if cs else 6
    crowdedness_score = min(max(crowdedness_score, 1), 10)

    # feature flags
    quarantine_enabled = input("Enable quarantine (y/n) [y]: ").strip() or "y"
    seasonality_enabled = input("Enable seasonality (y/n) [y]: ").strip() or "y"
    interventions_enabled = input("Enable interventions (y/n) [y]: ").strip() or "y"
    multi_wave = input("Enable multi-wave (y/n) [n]: ").strip() or "n"

    # vaccination
    vaccination_enabled = input("Enable vaccination (y/n) [n]: ").strip() or "n"
    daily_vaccination_rate = input("Daily vaccination rate (fraction, e.g. 0.01 for 1%) [0.0]: ").strip()
    daily_vaccination_rate = float(daily_vaccination_rate) if daily_vaccination_rate else 0.0

    # incubation period
    incubation_period = input(f"Incubation period (days) [{INCUBATION_PERIOD_DEFAULT}]: ").strip()
    incubation_period = int(incubation_period) if incubation_period else INCUBATION_PERIOD_DEFAULT

    # Unified variant/multi-wave
    num_waves = input("How many variant/multi-wave events? [1]: ").strip()
    num_waves = int(num_waves) if num_waves else 1
    waves = []
    for i in range(num_waves):
        print(f"--- Wave {i+1} ---")
        wave_day = input(f"  Day of wave [e.g., {60 + i*30}]: ").strip()
        wave_day = int(wave_day) if wave_day else (60 + i*30)
        wave_beta = input("  Beta multiplier [e.g., 2.5 for dramatic wave]: ").strip()
        wave_beta = float(wave_beta) if wave_beta else 2.5  # More dramatic default
        wave_seed = input("  Seed new exposed (number) [e.g., 100 for dramatic wave]: ").strip()
        wave_seed = int(wave_seed) if wave_seed else 100  # More dramatic default
        waves.append({"day": wave_day, "beta": wave_beta, "seed": wave_seed})

    # testing intensity
    testing_rate = input("Testing rate (low/medium/high) [medium]: ").strip() or "medium"

    # mask decay
    mask_decay_rate = input(f"Mask decay rate per day (fraction, e.g. 0.01 for 1%) [{MASK_DECAY_RATE_DEFAULT}]: ").strip()
    mask_decay_rate = float(mask_decay_rate) if mask_decay_rate else MASK_DECAY_RATE_DEFAULT

    # travel between cities
    travel_enabled = input("Enable travel between cities (y/n) [n]: ").strip() or "n"
    travel_max = 0
    if yn_to_bool(travel_enabled):
        travel_max_in = input("Travel (number of new exposed imported per day, 1–10) [3]: ").strip()
        travel_max = int(travel_max_in) if travel_max_in else 3

    # reporting range
    rpmin = input("Reporting probability min [0.6]: ").strip()
    reporting_prob_min = float(rpmin) if rpmin else 0.6
    rpmax = input("Reporting probability max [0.85]: ").strip()
    reporting_prob_max = float(rpmax) if rpmax else 0.85
    if reporting_prob_max <= reporting_prob_min:
        print("Max must be > min; adjusting.")
        reporting_prob_max = min(0.95, reporting_prob_min + 0.1)

    # seed
    seed_in = input("Random seed [auto]: ").strip()
    random_seed = None if not seed_in or seed_in.lower() == "auto" else int(seed_in)

    params = {
        "population": population,
        "days": days,
        "initial_infected": initial_infected,
        "mask_score": mask_score,
        "crowdedness_score": crowdedness_score,
        "quarantine_enabled": quarantine_enabled,
        "seasonality_enabled": seasonality_enabled,
        "interventions_enabled": interventions_enabled,
        "reporting_prob_min": reporting_prob_min,
        "reporting_prob_max": reporting_prob_max,
        "multi_wave": multi_wave,
        "random_seed": random_seed,
        "vaccination_enabled": vaccination_enabled,
        "daily_vaccination_rate": daily_vaccination_rate,
        "incubation_period": incubation_period,
        "waves": waves,
        "testing_rate": testing_rate,
        "mask_decay_rate": mask_decay_rate,
        "travel_enabled": travel_enabled,
        "travel_max": travel_max,
        "mode": "interactive"
    }
    return params


def generate_random_inputs(batch_fixed=None):
    """
    Generate a random but realistic parameter config.
    Returns a dictionary of parameters.
    """
    population = random.randint(10_000, 1_000_000)
    if batch_fixed is None:
        days = random.randint(90, 365)
        max_init = max(1, min(population // 20, 5000))
        initial_infected = random.randint(10, max_init)

        mask_score = random.randint(1, 10)
        crowdedness_score = random.randint(1, 10)

        quarantine_enabled = random.choice(["y", "n"])
        seasonality_enabled = random.choice(["y", "n"])
        interventions_enabled = random.choice(["y", "n"])
        multi_wave = random.choice(["y", "n"])

        vaccination_enabled = random.choice(["y", "n"])
        daily_vaccination_rate = round(random.uniform(0.0, 0.02), 3)  # up to 2% per day
        incubation_period = random.randint(2, 7)
        testing_rate = random.choice(["low", "medium", "high"])
        mask_decay_rate = round(random.uniform(0.005, 0.02), 4)

        travel_enabled = random.choice(["y", "n"])
        travel_max = random.randint(1, 10) if yn_to_bool(travel_enabled) else 0

        low = round(random.uniform(0.3, 0.6), 2)
        high = round(random.uniform(low + 0.1, 0.95), 2)

        random_seed = random.randint(1, 999999)

        # Unified variant/multi-wave
        num_waves = random.randint(1, 2)
        waves = []
        for i in range(num_waves):
            wave_day = random.randint(30 + i*30, 120 + i*30)
            wave_beta = round(random.uniform(1.2, 2.5), 2)
            wave_seed = random.randint(20, 100)
            waves.append({"day": wave_day, "beta": wave_beta, "seed": wave_seed})
    else:
        days = batch_fixed.get('days') if batch_fixed.get('days') is not None else random.randint(90, 365)
        max_init = max(1, min(population // 20, 5000))
        initial_infected = random.randint(10, max_init)

        mask_score = batch_fixed.get('mask_score') if batch_fixed.get('mask_score') is not None else random.randint(1, 10)
        crowdedness_score = random.randint(1, 10)

        quarantine_enabled = random.choice(["y", "n"])
        seasonality_enabled = random.choice(["y", "n"])
        interventions_enabled = random.choice(["y", "n"])
        multi_wave = random.choice(["y", "n"])

        vaccination_enabled = random.choice(["y", "n"])
        daily_vaccination_rate = round(random.uniform(0.0, 0.02), 3)  # up to 2% per day
        incubation_period = random.randint(2, 7)
        testing_rate = random.choice(["low", "medium", "high"])
        mask_decay_rate = round(random.uniform(0.005, 0.02), 4)

        travel_enabled = random.choice(["y", "n"])
        travel_max = random.randint(1, 10) if yn_to_bool(travel_enabled) else 0

        low = round(random.uniform(0.3, 0.6), 2)
        high = round(random.uniform(low + 0.1, 0.95), 2)

        random_seed = random.randint(1, 999999)

        # Unified variant/multi-wave
        num_waves = batch_fixed.get('multi_wave_count') if batch_fixed.get('multi_wave_count') is not None else random.randint(1, 2)
        wave_days = [int(x) for x in batch_fixed.get('wave_days', '').split(',')] if batch_fixed.get('wave_days') else [60 + i*30 for i in range(num_waves)]
        wave_betas = [float(x) for x in batch_fixed.get('wave_betas', '').split(',')] if batch_fixed.get('wave_betas') else [2.5]*num_waves
        wave_seeds = [int(x) for x in batch_fixed.get('wave_seeds', '').split(',')] if batch_fixed.get('wave_seeds') else [100]*num_waves
        waves = []
        for i in range(num_waves):
            waves.append({
                'day': wave_days[i] if i < len(wave_days) else 60 + i*30,
                'beta': wave_betas[i] if i < len(wave_betas) else 2.5,
                'seed': wave_seeds[i] if i < len(wave_seeds) else 100
            })

    params = {
        "population": population,
        "days": days,
        "initial_infected": initial_infected,
        "mask_score": mask_score,
        "crowdedness_score": crowdedness_score,
        "quarantine_enabled": quarantine_enabled,
        "seasonality_enabled": seasonality_enabled,
        "interventions_enabled": interventions_enabled,
        "reporting_prob_min": low,
        "reporting_prob_max": high,
        "multi_wave": multi_wave,
        "random_seed": random_seed,
        "vaccination_enabled": vaccination_enabled,
        "daily_vaccination_rate": daily_vaccination_rate,
        "incubation_period": incubation_period,
        "waves": waves,
        "testing_rate": testing_rate,
        "mask_decay_rate": mask_decay_rate,
        "travel_enabled": travel_enabled,
        "travel_max": travel_max,
        "mode": "random"
    }
    return params 