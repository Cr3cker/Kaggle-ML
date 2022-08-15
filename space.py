import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.feature_selection import SequentialFeatureSelector
from catboost import CatBoostClassifier


pd.options.mode.chained_assignment = None

# Read data and set new index

X_full = pd.read_csv('/Users/mila/Desktop/Data-Science/spaceship-titanic/train.csv')
X_test_full = pd.read_csv('/Users/mila/Desktop/Data-Science/spaceship-titanic/test.csv')

X_full.set_index('PassengerId', inplace=True)
X_test_full.set_index('PassengerId', inplace=True)

# Imputing NaN values

X_full[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = X_full[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)
X_test_full[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = X_test_full[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)

X_full['Age'] = X_full['Age'].fillna(X_full['Age'].median())
X_test_full['Age'] = X_test_full['Age'].fillna(X_test_full['Age'].median())

X_full['VIP'] = X_full['VIP'].fillna(False)
X_test_full['VIP'] = X_test_full['VIP'].fillna(False)

X_full['HomePlanet'] = X_full['HomePlanet'].fillna('Mars')
X_test_full['HomePlanet'] = X_test_full['HomePlanet'].fillna('Mars')

X_full['Destination'] = X_full['Destination'].fillna("PSO J318.5-22")
X_test_full['Destination'] = X_test_full['Destination'].fillna("PSO J318.5-22")

X_full['CryoSleep'] = X_full['CryoSleep'].fillna(False)
X_test_full['CryoSleep'] = X_test_full['CryoSleep'].fillna(False)

X_full['Cabin'] = X_full['Cabin'].fillna('T/0/P')
X_test_full['Cabin'] = X_test_full['Cabin'].fillna('T/0/P')

X_full[['deck','num','side']] = X_full.Cabin.str.split('/', expand=True)
X_test_full[['deck','num','side']] = X_test_full.Cabin.str.split('/', expand=True)

# Adding total spendings column

total = []

for x in range(0, 8693):
    transit = []
    for i in X_full[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]:
        transit.append(X_full[i][x].tolist())
    total.append(sum(transit))
        
X_full['total'] = [value for value in total]

total = []

for x in range(0, 4277):
    transit = []
    for i in X_test_full[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]:
        transit.append(X_test_full[i][x].tolist())
    total.append(sum(transit))

X_test_full['total'] = [value for value in total]

# Dividing people by age

X_full['AgeGroup'] = 0
for i in range(6):
    X_full.loc[(X_full.Age >= 10 * i) & (X_full.Age < 10 * (i + 1)), 'AgeGroup'] = i

X_test_full['AgeGroup'] = 0
for i in range(6):
    X_test_full.loc[(X_test_full.Age >= 10 * i) & (X_test_full.Age < 10 * (i + 1)), 'AgeGroup'] = i

# Encoding categorical columns

categorical_cols = ['HomePlanet', 'VIP', 'CryoSleep', 'Destination', 'deck', 'side', 'num']
for i in categorical_cols:
    le = LabelEncoder()
    arr = np.concatenate((X_full[i], X_test_full[i])).astype(str)
    le.fit(arr)
    X_full[i] = le.transform(X_full[i].astype(str))
    X_test_full[i] = le.transform(X_test_full[i].astype(str))

# Dropping unnecessary  columns

X_full.drop(['Name', 'Cabin'], axis=1, inplace=True)
X_test_full.drop(['Name', 'Cabin'], axis=1, inplace=True)

X_full['Transported'] = X_full['Transported'].replace({True: 1, False: 0})

X = X_full.drop('Transported', axis=1)
y = X_full['Transported']

# Separating data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0) # You can use it to validate your data

# Creating a model to find the best features

model = CatBoostClassifier(verbose=False)
sf = SequentialFeatureSelector(model, scoring='accuracy', direction='backward')

sf.fit(X, y)

best_features = list(sf.get_feature_names_out())

# Creating a final model to predict 'Transported' column

model_final = CatBoostClassifier(eval_metric='Accuracy', verbose=0)

model_final.fit(X[best_features], y)

y_pred = model_final.predict(X_test_full[best_features])

# Generating a submission.csv file

sub = pd.DataFrame({'Transported': y_pred.astype(bool)}, index=X_test_full.index)
sub.to_csv('submission.csv')
