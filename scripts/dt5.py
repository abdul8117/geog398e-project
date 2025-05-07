import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

path = '/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi-2/cleaned_V2.0_status_intensity_observation_data.csv'
df = pd.read_csv(path)

#make new df for ground truth information taking just from 2023 from observation date which is year-month-day
df_gt = df[df['Observation_Date'].str.contains('2023')]

#make a new df for training usign the observation date to only get years 2011 - 2022
df_train = df[df['Observation_Date'].str.contains('2011|2012|2013|2014|2015|2016|2017|2018|2019|2020|2021|2022')]

# Define features (X) and target variable (y) for training
X_train = df_train[['AGDD', 'Daylength', 'Prcp', 'Tmax', 'Tmin', 'Year', 'Month', 'Day', 'Accum_Prcp', 'Species_ID']]  # Replace with your actual feature columns
y_train = df_train['Intensity_Value']
# Define features (X) and target variable (y) for testing
X_test = df_gt[['AGDD', 'Daylength', 'Prcp', 'Tmax', 'Tmin', 'Year', 'Month', 'Day', 'Accum_Prcp', 'Species_ID']]  # Replace with your actual feature columns
y_test = df_gt['Intensity_Value']
# Use X_train, X_test, y_train, y_test directly in train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size=0.5, random_state=5)
print(Xtrain.shape, Xtest.shape)



#Example 1 Decision tree with one hot encoded landcover

#This will take the information from the training data set and create a decision tree classifier

# Create a OneHotEncoder object
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit the encoder on the 'land_cover_type' column of the training data
# and transform both training and testing data
encoded_land_cover_train = encoder.fit_transform(df_train[['land_cover_type']])
encoded_land_cover_test = encoder.transform(df_gt[['land_cover_type']])

# Create column names for the encoded features
encoded_feature_names = encoder.get_feature_names_out(['land_cover_type'])

# Convert the encoded features to DataFrames
encoded_land_cover_train_df = pd.DataFrame(encoded_land_cover_train, columns=encoded_feature_names, index=df_train.index)
encoded_land_cover_test_df = pd.DataFrame(encoded_land_cover_test, columns=encoded_feature_names, index=df_gt.index)

# Concatenate the encoded features with the other features
X_train = pd.concat([df_train[['AGDD', 'Daylength', 'Phenophase_Definition_ID', 'Prcp', 'Tmax', 'Tmin', 'Species_ID']], encoded_land_cover_train_df], axis=1)
X_test = pd.concat([df_gt[['AGDD', 'Daylength', 'Phenophase_Definition_ID', 'Prcp', 'Tmax', 'Tmin','Species_ID']], encoded_land_cover_test_df], axis=1)

# Now you can train your Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 100, random_state=42)
dt_classifier.fit(X_train, y_train)

#Accuracy and prediction on decision tree 1
# Make predictions on the training set and the test set (test data being df_gt (2023))
y_train_pred = dt_classifier.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Decision Tree Training Accuracy: {train_accuracy}")

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Test Accuracy: {accuracy}")

#Example 1 random forest
from sklearn.ensemble import RandomForestClassifier
# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=20, max_features='sqrt',
                                       min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_classifier.fit(X_train, y_train)

y_train_pred = rf_classifier.predict(X_train)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Random Forest Training Accuracy: {train_accuracy}")

y_pred = rf_classifier.predict(X_test)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Testing Accuracy: {test_accuracy}")

