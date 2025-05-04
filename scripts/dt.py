import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree

path = r'C:\Users\abdul\OneDrive\Documents\GEOG398E Project\Project data\cleaned_V2.0_status_intensity_observation_data.csv'
df = pd.read_csv(path)

# Split into training (2011-2022) and testing (2023)
df_train = df[df['Observation_Date'].str.contains('2011|2012|2013|2014|2015|2016|2017|2018|2019|')]
df_gt = df[df['Observation_Date'].str.contains('2020|2020|2022|2023')]

# One-hot encode 'land_cover_type'
land_cover_dummies = pd.get_dummies(df['land_cover_type'], prefix='land_cover')

# Define features (X) and target (y)
X_train = pd.concat([
    df_train[['AGDD', 'Daylength', 'Prcp', 'Tmax', 'Tmin']],
    land_cover_dummies.loc[df_train.index]
], axis=1)

X_test = pd.concat([
    df_gt[['AGDD', 'Daylength', 'Prcp', 'Tmax', 'Tmin']],
    land_cover_dummies.loc[df_gt.index]
], axis=1)

y_train = df_train['Phenophase_Category']
y_test = df_gt['Phenophase_Category']

# Train Decision Tree
myDT = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
myDT.fit(X_train, y_train)

# Evaluate
ypred_train = myDT.predict(X_train)
ypred_test = myDT.predict(X_test)

print('Training accuracy:', accuracy_score(y_train, ypred_train))
print('Test accuracy:', accuracy_score(y_test, ypred_test))