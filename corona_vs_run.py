import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the Excel file
file_path = r"C:\Users\rishu narwal\Desktop\SVM_FDE\datasets\batch_20250724_045930\run_001\reported_cases.csv.xlsx"
df = pd.read_excel(file_path)

# Scale Reported_India to match the scale of Reported_Cases
india_scaled = df['Reported_India'] * (df['Reported_Cases'].max() / df['Reported_India'].max())
df['Reported_India_Scaled'] = india_scaled

# Seaborn setup
sns.set(style="darkgrid", palette="muted")

# Bilinear plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Day', y='Reported_Cases', label='Simulated (BIGBOY1.2)')
sns.lineplot(data=df, x='Day', y='Reported_India_Scaled', label='Reported India (Scaled)')
plt.title('Comparison of Synthetic vs Real Cases')
plt.xlabel('Day')
plt.ylabel('Reported Cases (Scaled)')
plt.legend()
plt.tight_layout()
plt.show()

# --------- Comparative Metrics ---------

# Peak infection day
peak_day_sim = df['Day'][df['Reported_Cases'].idxmax()]
peak_day_real = df['Day'][df['Reported_India'].idxmax()]

# Epidemic duration (days from 10% to 90% of peak)
def epidemic_duration(series):
    peak = series.max()
    above_10 = series >= 0.1 * peak
    above_90 = series >= 0.9 * peak
    try:
        start = df['Day'][above_10.idxmax()]
        end = df['Day'][len(above_10) - above_10[::-1].idxmax() - 1]
        return end - start
    except:
        return np.nan

duration_sim = epidemic_duration(df['Reported_Cases'])
duration_real = epidemic_duration(df['Reported_India'])

# Basic reproduction number approximation (peak growth slope)
def estimate_r0(series):
    growth = series.diff().rolling(3).mean()  # smoothing
    max_slope = growth.max()
    return round(1 + max_slope / (series.mean() + 1e-5), 2)

r0_sim = estimate_r0(df['Reported_Cases'])
r0_real = estimate_r0(df['Reported_India'])

# Slope of decline (average % drop per day after peak)
def decline_slope(series):
    peak_idx = series.idxmax()
    post_peak = series[peak_idx:]
    declines = -post_peak.diff()
    percent_drops = declines / post_peak.shift(1).replace(0, np.nan)
    return round(percent_drops.mean() * 100, 2)

slope_sim = decline_slope(df['Reported_Cases'])
slope_real = decline_slope(df['Reported_India'])

# Print comparisons
print(" Comparison Metrics:")
print(f"→ Peak Infection Day: BIGBOY1.2 = Day {peak_day_sim}, Real = Day {peak_day_real}")
print(f"→ Epidemic Duration: BIGBOY1.2 = {duration_sim} days, Real = {duration_real} days")
print(f"→ Estimated R₀: BIGBOY1.2 = {r0_sim}, Real = {r0_real}")
print(f"→ Decline Slope: BIGBOY1.2 = {slope_sim}%/day, Real = {slope_real}%/day")
