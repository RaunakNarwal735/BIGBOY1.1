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
MASK_MAX_REDUCTION = 0.6       # mask_score=10 -> -60% beta
CROWD_MAX_INCREASE = 0.7       # crowd_score=10 -> +70% beta

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