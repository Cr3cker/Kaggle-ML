from sklearn.ensemble import RandomForestClassifier
import math

# You can use GridSearchCV if you want to find the best hyperparams
# Here is the simple example:

# from sklearn.model_selection import GridSearchCV

# model_vip = RandomForestClassifier(random_state=0)
                                   
# params = {'n_estimators': range (50, 250, 10),
#           'max_depth': range (1, 40, 2),
#           'min_samples_leaf': range (1, 15),
#           'min_samples_split': range (2, 14, 2)}

# grid = GridSearchCV(model_vip, params, cv=5)
# grid.fit(X_full_vip.drop(['PassengerId'], axis=1), y_vip)

# After that, use grid.best_params_ to show the output 

# Creating models for vip and nvip
model_vip = RandomForestClassifier(n_estimators = 200, 
                                   max_depth = 30, 
                                   max_features = 4, 
                                   min_samples_split = 10, 
                                   min_samples_leaf = 10, 
                                   random_state = 0)

model.fit(X_full_vip.drop(['PassengerId'], axis=1), y_vip)
prediction_1 = model.predict(X_test_vip.drop(['PassengerId'], axis=1))

model_nvip = RandomForestClassifier(n_estimators = 200,
                                    max_depth = 30,
                                    max_features = 4,
                                    min_samples_split = 10,
                                    min_samples_leaf = 10,
                                    random_state = 0)

model_2.fit(X_full_nvip.drop(['PassengerId'], axis=1), y_nvip)
prediction_2 = model_2.predict(X_test_nvip.drop(['PassengerId'], axis=1))

# Generating the merge file of predictions to submit them to Kaggle
X_test_vip['pred_1'] = prediction_1
X_test_nvip['pred_2'] = prediction_2

X_merge = X_test_full.merge(X_test_vip['pred_1'], left_index=True, right_index=True, how='left')
X_merge = X_merge.merge(X_test_nvip['pred_2'], left_index=True, right_index=True, how='left')
X_merge['pred'] = X_merge.apply(lambda x: x['pred_1'] if not math.isnan(x['pred_1']) else x['pred_2'], axis=1)

# Generating the final output.csv file
output_1 = pd.DataFrame({'PassengerId': X_merge.PassengerId,
                         'Transported': X_merge['pred']})
output_1.to_csv('submission.csv', index=False)
