import pandas as pd
import matplotlib.pyplot as plt
import json

with open('intensity_mapping.json') as f:
    intensity_mapping = json.load(f)

df = pd.read_csv("datasets/datasets-for-roi/cleaned_status_intensity_observation_data.csv")

df["Intensity_Value"] = df["Intensity_Value"].map(intensity_mapping)

print(df.head(10))
print(df.shape)

df['Observation_Date'] = pd.to_datetime(df['Observation_Date'], errors='coerce')

# Drop rows where the date couldn't be parsed
df = df.dropna(subset=['Observation_Date'])

# Extract the month from the date
df['Month'] = df['Observation_Date'].dt.month

# Count total number of observations per month
monthly_counts = df['Month'].value_counts().sort_index()

# Print the result
print("Total observations per month:")
print(monthly_counts)

plt.figure(figsize=(10, 5))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Total Observations Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Observations')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


