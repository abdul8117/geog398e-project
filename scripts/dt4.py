import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# 4. this produce decision tree visualizations and importance of each category 
# Load the dataset
path = '/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi-2/cleaned_V2.0_status_intensity_observation_data.csv'
df = pd.read_csv(path)

# Feature categorization functions
def categorize_temperature(temp):
    """Categorize temperature (Tmax or Tmin) into buckets"""
    if temp < 0:
        return "Below_Freezing"
    elif 0 <= temp < 10:
        return "Very_Cold"
    elif 10 <= temp < 20:
        return "Cold"
    elif 20 <= temp < 30:
        return "Moderate"
    elif 30 <= temp < 40:
        return "Hot"
    else:
        return "Very_Hot"

def categorize_precipitation(prcp):
    """Categorize precipitation amounts"""
    if prcp == 0:
        return "None"
    elif 0 < prcp < 1:
        return "Very_Light"
    elif 1 <= prcp < 5:
        return "Light"
    elif 5 <= prcp < 10:
        return "Moderate"
    elif 10 <= prcp < 20:
        return "Heavy"
    elif prcp >= 20:
        return "Very_Heavy"
    else:
        return "Unknown"  # For missing or invalid values like -9999

def categorize_accum_prcp(prcp):
    """Categorize accumulated precipitation"""
    if prcp < 100:
        return "Low"
    elif 100 <= prcp < 200:
        return "Moderate_Low"
    elif 200 <= prcp < 300:
        return "Moderate"
    elif 300 <= prcp < 400:
        return "Moderate_High"
    else:
        return "High"

def categorize_agdd(agdd):
    """Categorize accumulated growing degree days (AGDD)"""
    if agdd < 500:
        return "Very_Low"
    elif 500 <= agdd < 750:
        return "Low"
    elif 750 <= agdd < 1000:
        return "Moderate_Low"
    elif 1000 <= agdd < 1250:
        return "Moderate"
    elif 1250 <= agdd < 1500:
        return "Moderate_High"
    else:
        return "High"

def categorize_daylength(daylength):
    """Categorize daylength into buckets"""
    if daylength < 35000:
        return "Very_Short"
    elif 35000 <= daylength < 40000:
        return "Short"
    elif 40000 <= daylength < 45000:
        return "Moderate_Short"
    elif 45000 <= daylength < 50000:
        return "Moderate_Long"
    else:
        return "Long"

def categorize_season(month):
    """Convert month to season"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

# Apply categorization to the dataset
df['Tmax_Category'] = df['Tmax'].apply(categorize_temperature)
df['Tmin_Category'] = df['Tmin'].apply(categorize_temperature)
df['Prcp_Category'] = df['Prcp'].apply(categorize_precipitation)
df['Accum_Prcp_Category'] = df['Accum_Prcp'].apply(categorize_accum_prcp)
df['AGDD_Category'] = df['AGDD'].apply(categorize_agdd)
df['Daylength_Category'] = df['Daylength'].apply(categorize_daylength)
df['Season'] = df['Month'].apply(categorize_season)

# Group species IDs by popularity to reduce dimensionality
species_counts = df['Species_ID'].value_counts()
top_species = species_counts[species_counts > 100].index.tolist()  # Species with >100 observations

def categorize_species(species_id):
    """Group less common species together"""
    if species_id in top_species:
        return f"Species_{species_id}"
    else:
        return "Other_Species"

df['Species_Group'] = df['Species_ID'].apply(categorize_species)

# Extract year, month, day from Observation_Date if needed
# Check if the date is already split into Year, Month, Day
if 'Observation_Date' in df.columns:
    if 'Year' not in df.columns:
        df['Year'] = pd.to_datetime(df['Observation_Date']).dt.year
    if 'Month' not in df.columns:
        df['Month'] = pd.to_datetime(df['Observation_Date']).dt.month
    if 'Day' not in df.columns:
        df['Day'] = pd.to_datetime(df['Observation_Date']).dt.day

# Split into training (2011-2019) and testing (2020-2023)
df_train = df[df['Year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])]
df_test = df[df['Year'].isin([2020, 2021, 2022, 2023])]

# Create dummy variables for categorical features
categorical_features = [
    'Tmax_Category', 'Tmin_Category', 'Prcp_Category', 'Accum_Prcp_Category',
    'AGDD_Category', 'Daylength_Category', 'Season', 'Species_Group',
    'land_cover_type', 'Phenophase_Description'
]

# One-hot encoding for all categorical features
dummies = pd.get_dummies(df[categorical_features])

# Define features (X) and target (y)
numerical_features = ['Month', 'Day']

X_train = pd.concat([
    df_train[numerical_features],
    dummies.loc[df_train.index]
], axis=1)

X_test = pd.concat([
    df_test[numerical_features],
    dummies.loc[df_test.index]
], axis=1)

y_train = df_train['Intensity_Value']
y_test = df_test['Intensity_Value']

# Ensure X_train and X_test have the same columns
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

X_test = X_test[X_train.columns]

# Train Decision Tree
print("Training decision tree model...")
myDT = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
myDT.fit(X_train, y_train)

# Evaluate
ypred_train = myDT.predict(X_train)
ypred_test = myDT.predict(X_test)

print('Training accuracy:', accuracy_score(y_train, ypred_train))
print('Test accuracy:', accuracy_score(y_test, ypred_test))

# Print detailed classification report
print("\nClassification Report (Test Data):")
print(classification_report(y_test, ypred_test))

# Feature importance
feature_importances = pd.DataFrame(
    myDT.feature_importances_,
    index=X_train.columns,
    columns=['importance']
).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importances.head(20))

# Visualize tree (optional)
plt.figure(figsize=(20, 10))
tree.plot_tree(myDT, max_depth=3, feature_names=X_train.columns, filled=True)
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
print("Tree visualization saved as 'decision_tree_visualization.png'")