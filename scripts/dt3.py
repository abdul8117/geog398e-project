import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# 3. this use dt2's sketch and data analysis from notebook to better categorize temp, prcp, and also map phenophases alot better (have not done month yet)
# this produce decision tree visualizations and importance of each category 
# Training optimized decision tree...

# Training Accuracy: 0.4117 //why is my training accuracy low?
# Test Accuracy: 0.2605

# Load the dataset
path = '/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi-2/cleaned_V2.0_status_intensity_observation_data.csv'
df = pd.read_csv(path)

# --- Add Month extraction FIRST ---
if 'Observation_Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Observation_Date']).dt.year
    df['Month'] = pd.to_datetime(df['Observation_Date']).dt.month
    df['Day'] = pd.to_datetime(df['Observation_Date']).dt.day

# --- Add Growing_Month EARLY ---
df['Growing_Month'] = df['Month'].apply(lambda m: 1 if 4 <= m <= 9 else 0)

# Improved categorization functions
def categorize_temperature(temp):
    if temp < 5: return "Dormant"
    elif 5 <= temp < 10: return "Chilling"
    elif 10 <= temp < 15: return "Early_Growth"
    elif 15 <= temp < 25: return "Optimal"
    elif 25 <= temp < 35: return "Heat_Stress"
    else: return "Extreme_Heat"

def categorize_precipitation(prcp):
    if prcp == 0: return "None"
    elif 0 < prcp < 2.5: return "Light"
    elif 2.5 <= prcp < 7.6: return "Moderate"
    elif 7.6 <= prcp < 25.4: return "Heavy"
    else: return "Extreme"

def categorize_agdd(agdd):
    if agdd < 200: return "Dormant"
    elif 200 <= agdd < 500: return "Bud_Swell"
    elif 500 <= agdd < 1000: return "Leaf_Out"
    elif 1000 <= agdd < 1500: return "Flowering"
    else: return "Fruit_Senescence"

def categorize_daylength(daylength):
    hours = daylength / 3600
    if hours < 10: return "Winter"
    elif 10 <= hours < 12: return "Spring_Fall"
    elif 12 <= hours < 14: return "Growing_Season"
    else: return "Peak_Summer"

def categorize_accum_prcp(prcp):
    if prcp < 200: return "Dry"
    elif 200 <= prcp < 400: return "Normal"
    elif 400 <= prcp < 600: return "Moist"
    else: return "Saturated"

# --- THEN apply categorizations ---
df['Tmax_Category'] = df['Tmax'].apply(categorize_temperature)
df['Tmin_Category'] = df['Tmin'].apply(categorize_temperature)
df['Prcp_Category'] = df['Prcp'].apply(categorize_precipitation)
df['Accum_Prcp_Category'] = df['Accum_Prcp'].apply(categorize_accum_prcp)
df['AGDD_Category'] = df['AGDD'].apply(categorize_agdd)
df['Daylength_Category'] = df['Daylength'].apply(categorize_daylength)

# Enhanced land cover grouping
land_cover_map = {
    'Deciduous forest': 'Forest',
    'Evergreen forest': 'Forest',
    'Mixed forest': 'Forest',
    'Developed, open space': 'Urban',
    'Developed, low intensity': 'Urban',
    'Cultivated crops': 'Agriculture',
    'Woody wetlands': 'Wetlands',
    'Pasture/hay': 'Other'
}
df['Land_Cover_Group'] = df['land_cover_type'].map(land_cover_map)

# Phenophase consolidation
phenophase_map = {
    'Flowers or flower buds': 'Bud_Development',
    'Open flowers': 'Active_Flowering',
    'Pollen release (flowers)': 'Pollen_Release',
    'Pollen cones (conifers)': 'Conifer_Repro',
    'Open pollen cones (conifers)': 'Conifer_Repro',
    'Pollen release (conifers)': 'Conifer_Repro',
    'Open flowers (grasses/sedges)': 'Grass_Flowering'
}
df['Phenophase_Group'] = df['Phenophase_Description'].map(phenophase_map)

# Functional species grouping
def categorize_species(species_id):
    deciduous = {3, 12, 82}
    conifers = {7, 1172}
    if species_id in deciduous: return 'Deciduous_Tree'
    elif species_id in conifers: return 'Conifer'
    else: return 'Other_Plants'
df['Species_Group'] = df['Species_ID'].apply(categorize_species)

# --- Now split the data ---
# 2011-2022 is all train
# 2023 is testing data
df_train = df[df['Year'] <= 2022]
df_test = df[df['Year'] >= 2023]


# Updated categorical features
categorical_features = [
    'Tmax_Category', 'Tmin_Category', 'Prcp_Category',
    'Accum_Prcp_Category', 'AGDD_Category', 'Daylength_Category',
    'Land_Cover_Group', 'Phenophase_Group', 'Species_Group'
]

# One-hot encoding
dummies = pd.get_dummies(df[categorical_features])

# Final feature selection
numerical_features = ['Growing_Month']
X_train = pd.concat([df_train[numerical_features], dummies.loc[df_train.index]], axis=1)
X_test = pd.concat([df_test[numerical_features], dummies.loc[df_test.index]], axis=1)

y_train = df_train['Intensity_Value']
y_test = df_test['Intensity_Value']

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Train decision tree
print("Training optimized decision tree...")
myDT = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=0.8
)
myDT.fit(X_train, y_train)

# Evaluation
ypred_train = myDT.predict(X_train)
ypred_test = myDT.predict(X_test)

print(f'\nTraining Accuracy: {accuracy_score(y_train, ypred_train):.4f}')
print(f'Test Accuracy: {accuracy_score(y_test, ypred_test):.4f}')

print("\nDetailed Classification Report:")
print(classification_report(y_test, ypred_test))

# Feature importance
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': myDT.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Predictive Features:")
print(feature_importances.head(15))

# Visualize tree
plt.figure(figsize=(20,12))
class_names = [str(c) for c in sorted(y_train.unique())]
tree.plot_tree(myDT, max_depth=3, feature_names=X_train.columns,
               class_names=class_names, filled=True)
plt.savefig('optimized_tree.png', dpi=300)
print("\nTree visualization saved as optimized_tree.png")
