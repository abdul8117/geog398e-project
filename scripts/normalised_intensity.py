import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/datasets-for-roi/cleaned_status_intensity_observation_data.csv")

# Filter only rows with valid Intensity_Value
df_intensity = df[df['Intensity_Value'].notna()].copy()

# Group by Day_of_Year and Intensity_Value
daily_counts = df_intensity.groupby(['Day_of_Year', 'Intensity_Value']).size().unstack(fill_value=0)

# Total observations per day
daily_totals = daily_counts.sum(axis=1)

# Normalize: divide each intensity level count by total for that day
daily_normalized = daily_counts.div(daily_totals, axis=0)

# Optional: fill missing intensity levels with 0 for clean plotting
for level in ['low', 'high']:
    if level not in daily_normalized.columns:
        daily_normalized[level] = 0


plt.figure(figsize=(12, 6))

plt.plot(daily_normalized.index, daily_normalized['high'], color='red', label='High')
# plt.plot(daily_normalized.index, daily_normalized['med'], color='yellow', label='Medium')
plt.plot(daily_normalized.index, daily_normalized['low'], color='green', label='Low')

plt.title('Proportion of Pollen Intensity Observations Throughout the Year')
plt.xlabel('Day of Year')
plt.ylabel('Proportion of Observations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
