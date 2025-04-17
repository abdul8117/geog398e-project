import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/datasets-for-roi/cleaned_status_intensity_observation_data.csv")

print(df.head(10))
print(df.shape)


# Count number of observations for each intensity per day
intensity_counts = df[df['Intensity_Value'].notna()].groupby(['Day_of_Year', 'Intensity_Value']).size().unstack(fill_value=0)

# Sort by day of year
intensity_counts = intensity_counts.sort_index()

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(intensity_counts.index, intensity_counts.get('high', []), color='red', label='High')
# plt.plot(intensity_counts.index, intensity_counts.get('med', []), color='yellow', label='Medium')
plt.plot(intensity_counts.index, intensity_counts.get('low', []), color='green', label='Low')

# Customize the plot
plt.title('Pollen Intensity Observations Throughout the Year')
plt.xlabel('Day of Year')
plt.ylabel('Number of Observations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
