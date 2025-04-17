import pandas as pd
import json

# file_path = "datasets/status_intensity_observation_data.csv"
file_path = "datasets/datasets-for-roi/status_intensity_observation_data.csv"

df = pd.read_csv(file_path)

df = df[df['Intensity_Value'] != '-9999']

number_of_rows = len(df)

df['Intensity_Value'] = df['Intensity_Value'].astype(str)

number_of_rows = len(df)
unique_intensity_values = sorted(df.Intensity_Value.unique())

print("Number of usable rows:", number_of_rows)

print("Below are the types of reporting methods on the intensity of pollen")
print(*unique_intensity_values, sep="\n")

cols = ['Update_Datetime', 'Site_ID', 'Elevation_in_Meters', 'Genus', 'Species', 'Common_Name', 'Kingdom', 'Phenophase_Description', 'Phenophase_Status', 'Abundance_Value']
df = df.drop(columns=cols)

with open('intensity_mapping.json') as f:
    intensity_mapping = json.load(f)

df["Intensity_Value"] = df["Intensity_Value"].map(intensity_mapping)

print(df.head(10))
print(df.shape)

# save cleaned dataset
df.to_csv("datasets/datasets-for-roi/cleaned_status_intensity_observation_data.csv", index=False)