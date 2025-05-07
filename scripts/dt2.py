import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree

#2. This is a scratch sheet for dt3!!! please look into dt3
path = '/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi-2/cleaned_V2.0_status_intensity_observation_data.csv'
df = pd.read_csv(path)

# Year       Count     
# --------------------
# 2011       1         
# 2012       41        
# 2013       851       
# 2014       958       
# 2015       782       
# 2016       1717      
# 2017       2270      
# 2018       2287      
# 2019       2211      
# 2020       1701      
# 2021       3256      
# 2022       5019      
# 2023       5738      <--- (maybe later but let's predict 2024 and 2025 first )
# 2024       3659      <--- prediciting this (ok now we figure out that this have -9999 in the rows so we dropped everything here)
# 2025       1705      <--- prediciting this (ok now we figure out that this have -9999 in the rows so we dropped everything here)
# final data sets 32196 rows.

# Split into training (2011-2022) and testing (2023)
df_train = df[df['Observation_Date'].str.contains('2011|2012|2013|2014|2015|2016|2017|2018|2019|')] #seems like soemthing wrong over here
df_gt = df[df['Observation_Date'].str.contains('2020|2020|2022|2023')] #seems like soemthing wrong over here


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

# NEED TO CATEGORIZE
# 'Species_ID', (different number id, but idk if it should be encoded some more, 301 unqiue species ID, something like  82,61, 3....etc (301 different) )
# 'Phenophase_Description',  One-hot encode *** 
# 'Observation_Date', *** (split it into year, month, day (maybe day is not needed))
# 'Intensity_Value', <----- *** ----- prediciting this and it's split into 0.0-9.0 intensity 
        #     # Count-based categories
        #     "Less than 3": 0, 
        #     "3 to 10": 1,
        #     "11 to 100": 2,
        #     "101 to 1,000": 4,
        #     "1,001 to 10,000": 6,
        #     "More than 10,000": 8,
        #     "More than 10": 2,
        #     "More than 1,000": 5,
            
        #     # Percentage-based categories
        #     "Less than 5%": 0,
        #     "5-24%": 1,
        #     "25-49%": 3,
        #     "50-74%": 5,
        #     "75-94%": 7,
        #     "95% or more": 9,
            
        #     # Qualitative categories
        #     "Little": 1,
        #     "Some": 3,
        #     "Lots": 6,
        #     "Peak flower": 8,
        #     "Peak opening": 8,
        #     "Peak pollen": 8
        # }
# 'AGDD', (what is this again?)
# 'Tmax',     
    # if 0 <= temp <= 35: return "Low"
    # elif 36 <= temp <= 45: return "Medium"
    # elif 45 < temp <= 90: return "High"
    # return "Unknown"
# 'Tmin',  (maybe something about negative tempeatures)
    # if 0 <= temp <= 35: return "Low"
    # elif 36 <= temp <= 45: return "Medium"
    # elif 45 < temp <= 90: return "High"
    # return "Unknown"
# 'Prcp',
# 'Accum_Prcp', (what is the differences between prcp vs accum_prcp?)
  # def categorize_precipitation(prcp):
  #     if 0 <= prcp <= 0.3: return "High"
  #     elif 0.4 <= prcp <= 1: return "Medium"
  #     elif 1.1 <= prcp <= 10: return "Low"
  #     return "Unknown"

  # def categorize_prcp_amount(prcp):
  #     if 0 <= prcp <= 2: return "High"
  #     elif 2 < prcp <= 4: return "Medium"
  #     return "Low"
# 'Daylength', 
# 'land_cover_type' One-hot encode

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