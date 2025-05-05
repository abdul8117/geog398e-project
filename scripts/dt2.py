import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree

path = '/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi-2/cleaned_V2.0_status_intensity_observation_data.csv'
df = pd.read_csv(path)

# Split into training (2011-2022) and testing (2023)
df_train = df[df['Observation_Date'].str.contains('2011|2012|2013|2014|2015|2016|2017|2018|2019|')]
df_gt = df[df['Observation_Date'].str.contains('2020|2020|2022|2023')]

# One-hot encode 'land_cover_type'
land_cover_dummies = pd.get_dummies(df['land_cover_type'], prefix='land_cover')
phenophase_dummies = pd.get_dummies(df['Phenophase_Description'], prefix='phenophase')

# Combine all dummy variables into a single dataframe
dummies = pd.concat([land_cover_dummies, phenophase_dummies], axis=1)


# Define features (X) and target (y)
X_train = pd.concat([
    df_train[['AGDD', 'Daylength', 'Prcp', 'Tmax', 'Tmin', 'Year', 'Month', 'Day', 'Accum_Prcp', 'Species_ID', ]],
    dummies.loc[df_train.index]

], axis=1)

X_test = pd.concat([
    df_gt[['AGDD', 'Daylength', 'Prcp', 'Tmax', 'Tmin', 'Year', 'Month', 'Day', 'Accum_Prcp', 'Species_ID', ]],
    dummies.loc[df_gt.index]
], axis=1)

# ALL CATEGORY
# 'Observation_ID', 'Species_ID', 'Phenophase_ID', 'Phenophase_Category',
#        'Phenophase_Description', 'Phenophase_Definition_ID',
#        'Observation_Date', 'Day_of_Year', 'Intensity_Category_ID',
#        'Intensity_Value', 'Site_Visit_ID', 'AGDD', 'Tmax', 'Tmin', 'Prcp',
#        'Accum_Prcp', 'Daylength', 'county_fips', 'land_cover_type'],

# USED
# 'Observation_ID', 'Species_ID', 'Phenophase_ID', 'Phenophase_Category',
#   'Phenophase_Definition_ID',
#   'Day_of_Year', 
#     'Site_Visit_ID',  'Daylength', 'county_fips', '],
#'Intensity_Category_ID' (CAN'T USE THIS, other wise it will just figure it out),

y_train = df_train['Intensity_Value']
y_test = df_gt['Intensity_Value']

# Train Decision Tree
myDT = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
myDT.fit(X_train, y_train)

# Evaluate
ypred_train = myDT.predict(X_train)
ypred_test = myDT.predict(X_test)

print('Training accuracy:', accuracy_score(y_train, ypred_train))
print('Test accuracy:', accuracy_score(y_test, ypred_test))