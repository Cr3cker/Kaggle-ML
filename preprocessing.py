import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

pd.options.mode.chained_assignment = None


X_full = pd.read_csv('/Users/mila/Desktop/Data-Science/spaceship-titanic/train.csv')
X_test_full = pd.read_csv('/Users/mila/Desktop/Data-Science/spaceship-titanic/test.csv')

X_test_ids = X_test_full.PassengerId

y = X_full.Transported

dummies_cols = ['HomePlanet', 'Destination']

numerical_transformer = SimpleImputer(strategy='median')

# Dividing cols into different types
numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_full.columns if X_full[cname].dtype == 'object']

# Imputing numerical cols
X_full[numerical_cols] = pd.DataFrame(numerical_transformer.fit_transform(X_full[numerical_cols]))
X_test_full[numerical_cols] = pd.DataFrame(numerical_transformer.transform(X_test_full[numerical_cols]))

# Imputing categorical cols 
categorical_transformer = SimpleImputer(strategy='most_frequent')

X_full[categorical_cols] = pd.DataFrame(categorical_transformer.fit_transform(X_full[categorical_cols]))
X_test_full[categorical_cols] = pd.DataFrame(categorical_transformer.transform(X_test_full[categorical_cols]))

# Additional targets for VIP feature
y_vip = X_full[X_full['VIP'] == True]['Transported']
y_nvip = X_full[X_full['VIP'] == False]['Transported']

# Dropping unnecessary  columns
X_full.drop(['Transported', 'Name', 'Cabin'], axis=1, inplace=True)
X_test_full.drop(['Name', 'Cabin'], axis=1, inplace=True)

# Getting some dummies...
df_train = pd.get_dummies(X_full[dummies_cols])
df_test = pd.get_dummies(X_test_full[dummies_cols])

# Concatenating dummies with X_full and X_test_full
X_full = pd.concat([df_train, X_full], axis=1)
X_test_full = pd.concat([df_test, X_test_full], axis=1)

# Dropping dummies cols in the final dataset
X_full.drop([col for col in dummies_cols], axis=1, inplace=True)
X_test_full.drop([col for col in dummies_cols], axis=1, inplace=True)

# Creating vip and non-vip datasets 
X_full_vip = X_full[X_full['VIP'] == True]
X_full_nvip = X_full[X_full['VIP'] == False]

X_test_vip = X_test_full[X_test_full['VIP'] == True]
X_test_nvip = X_test_full[X_test_full['VIP'] == False]

# Dropping VIP column
X_full_vip.drop(['VIP'], axis=1, inplace=True)
X_full_nvip.drop(['VIP'], axis=1, inplace=True)

X_test_vip.drop(['VIP'], axis=1, inplace=True)
X_test_nvip.drop(['VIP'], axis=1, inplace=True)
