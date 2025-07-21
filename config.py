"""
Configuration module for epidemic dataset generator.
Contains all global constants and default parameters.
"""

import os

# =============================
# Output Directory
# =============================
BASE_SAVE_DIR = r"C:\Users\rishu narwal\Desktop\SVM_FDE\datasets"  # Default output directory

# =============================
# Behavioral Beta Scaling
# =============================
MASK_MAX_REDUCTION = 0.15       # mask_score=10 -> -15% beta
CROWD_MAX_INCREASE = 1.3       # crowd_score=10 -> +130% beta

# =============================
# Seasonality Defaults
# =============================
SEASONAL_AMP = 0.25            # +/- 25%
SEASONAL_PERIOD = 60           # 60-day cycle

# =============================
# Quarantine
# =============================
QUARANTINE_FRACTION = 0.5      # 50% of infected effectively isolated if enabled

# =============================
# Recovery Rate (gamma)
# =============================
GAMMA = 0.08                   # slower recovery than before -> longer epidemic

# =============================
# Central Peak Envelope (Gaussian)
# =============================
CENTER_AMP = 0.5               # up to +50% beta at mid-epidemic
CENTER_SIGMA_FRAC = 0.2        # width = fraction of total days (std dev)

# =============================
# Multi-wave/Variant Parameters
# =============================
MULTIWAVE_BETA_MULT = 1.5      # 50% beta increase during second wave
MULTIWAVE_DURATION = 20        # days
VARIANT_BETA_MULTIPLIER_DEFAULT = 1.5
VARIANT_DAY_DEFAULT = 60
VARIANT_WAVE_DURATION = 40     # days for a more dramatic, realistic wave

# =============================
# Intervention Window
# =============================
INT_DURATION = 14              # days
INT_MULTIPLIER = 0.4           # 60% beta reduction

# =============================
# Reporting Jitter
# =============================
REPORT_SIGMA_ABS = 0.05        # absolute +/- noise to reporting prob
REPORT_CLIP_MIN = 0.0
REPORT_CLIP_MAX = 0.99

# =============================
# Random Dropper
# =============================
DROP_MIN_INTERVAL = 1          # min days between drops
DROP_MAX_INTERVAL = 3          # max days between drops
DROP_MIN_CASES = 1
DROP_MAX_CASES = 17

# =============================
# Vaccination, Incubation, Mask Decay
# =============================
default_vaccination_rate = 0.0  # default: 0% per day
INCUBATION_PERIOD_DEFAULT = 4   # days
MASK_DECAY_RATE_DEFAULT = 0.01  # 1% per day 

# =============================
# User Parameter Scaling Factors
# =============================
# These control how strongly user input affects the simulation
MASK_SCORE_SCALING = 0.015         # mask_score * this (default: 0.06, so 10 = 0.6)
CROWD_SCORE_SCALING = 0.15        # (crowd_score-1) * this (default: 0.07, so 9 = 0.63)
QUARANTINE_EFFECT_SCALING = 0.5   # Fraction of infectious pool removed if quarantine enabled
VACCINATION_EFFECT_SCALING = 0.9  # Fraction of vaccinated who become immune (default: 0.9) 