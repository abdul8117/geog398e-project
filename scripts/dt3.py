import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report  # Updated import
from sklearn import tree
# ------ Load Data ------
path = '/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi-2/cleaned_V2.0_status_intensity_observation_data.csv'
df = pd.read_csv(path)

# ------ Data Splitting ------
# Convert Observation_Date to datetime and extract year
df['Year'] = pd.to_datetime(df['Observation_Date']).dt.year
df_train = df[(df['Year'] >= 2011) & (df['Year'] <= 2022)]
df_gt = df[df['Year'] >= 2020]  # Test on 2020-2023 data

# ------ Feature Engineering ------
def categorize_temperature(temp):
    if 0 <= temp <= 35: return "Low"
    elif 36 <= temp <= 45: return "Medium"
    elif 45 < temp <= 90: return "High"
    return "Unknown"

def categorize_precipitation(prcp):
    if 0 <= prcp <= 0.3: return "High"
    elif 0.4 <= prcp <= 1: return "Medium"
    elif 1.1 <= prcp <= 10: return "Low"
    return "Unknown"

def categorize_prcp_amount(prcp):
    if 0 <= prcp <= 2: return "High"
    elif 2 < prcp <= 4: return "Medium"
    return "Low"

# Apply categorization to both datasets
for df_set in [df_train, df_gt]:
    # Create categories
    df_set.loc[:, "Tmax_Category"] = df_set["Tmax"].apply(categorize_temperature)
    df_set.loc[:, "Tmin_Category"] = df_set["Tmin"].apply(categorize_temperature)
    df_set.loc[:, "Prcp_Category"] = df_set["Prcp"].apply(categorize_precipitation)
    df_set.loc[:, "Prcp_Amount_Category"] = df_set["Prcp"].apply(categorize_prcp_amount)
    
    # Convert to numerical values
    category_mapping = {"Low": 0, "Medium": 1, "High": 2, "Unknown": -1}
    df_set.loc[:, "Tmax_Value"] = df_set["Tmax_Category"].map(category_mapping)
    df_set.loc[:, "Tmin_Value"] = df_set["Tmin_Category"].map(category_mapping)
    df_set.loc[:, "Prcp_Value"] = df_set["Prcp_Category"].map(category_mapping)
    df_set.loc[:, "Prcp_Amount_Value"] = df_set["Prcp_Amount_Category"].map(category_mapping)

# ------ One-Hot Encoding ------
# Initialize encoders
land_cover_encoder = OneHotEncoder(handle_unknown='ignore')
phenophase_encoder = OneHotEncoder(handle_unknown='ignore')

# Fit on training data
land_cover_encoder.fit(df_train[['land_cover_type']])
phenophase_encoder.fit(df_train[['Phenophase_Description']])

# Transform both datasets
land_cover_train = land_cover_encoder.transform(df_train[['land_cover_type']])
land_cover_test = land_cover_encoder.transform(df_gt[['land_cover_type']])

phenophase_train = phenophase_encoder.transform(df_train[['Phenophase_Description']])
phenophase_test = phenophase_encoder.transform(df_gt[['Phenophase_Description']])

# ------ Feature Construction ------
base_features = [
    'AGDD', 'Daylength', 'Prcp', 'Tmax', 'Tmin', 'Year',
    'Month', 'Day', 'Accum_Prcp', 'Species_ID',
    'Tmax_Value', 'Tmin_Value', 'Prcp_Value', 'Prcp_Amount_Value'
]

# Create final feature matrices
X_train = pd.DataFrame(
    data=np.hstack([
        df_train[base_features].values,
        land_cover_train.toarray(),
        phenophase_train.toarray()
    ]),
    index=df_train.index
)

X_test = pd.DataFrame(
    data=np.hstack([
        df_gt[base_features].values,
        land_cover_test.toarray(),
        phenophase_test.toarray()
    ]),
    index=df_gt.index
)

# ------ Target Variables ------
y_train = df_train['Intensity_Value']
y_test = df_gt['Intensity_Value']

# ------ Model Training ------
myDT = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10
)
myDT.fit(X_train, y_train)

# ------ Evaluation ------
ypred_train = myDT.predict(X_train)
ypred_test = myDT.predict(X_test)

print('\nTraining Metrics:')
print('Accuracy:', accuracy_score(y_train, ypred_train))
print('Balanced Accuracy:', balanced_accuracy_score(y_train, ypred_train))

print('\nTest Metrics:')
print('Accuracy:', accuracy_score(y_test, ypred_test))
print('Balanced Accuracy:', balanced_accuracy_score(y_test, ypred_test))
print('\nClassification Report:')
print(classification_report(y_test, ypred_test))