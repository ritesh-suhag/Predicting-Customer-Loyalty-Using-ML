
##############################################################################
# RANDOM FOREST FOR REGRESSION - ABC GROCERY TASK
##############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORT REQUIRED PACKAGES ~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
# from sklearn.feature_selection import RFECV
# We wouldn't need the RFECV.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORT SAMPLE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# IMPORT

# We enter rb because we are reading a file in from pickle
data_for_model = pickle.load(open("data/abc_regression_modelling.p", "rb"))

# DROP UNECESSARY COLUMNS

# Dropiing customer id as we don't really need it.
data_for_model.drop(["customer_id"], axis = 1, inplace = True)

# SHUFFLE DATA 

# It's better to shuffle the data in case we didn't know about any ordering of the data done previously.
data_for_model = shuffle(data_for_model, random_state = 42)

# ~~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH MISSING VALUES ~~~~~~~~~~~~~~~~~~~~~~~~~

data_for_model.isna().sum()
# Since the number of NA are very low we can directly drop them.

# Dropping missing values - 
data_for_model.dropna(how = "any", inplace = True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH OUTLIERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# We don't need to remove outliers in case of decision trees or random forest.

# ~~~~~~~~~~~~~~~~~~~~~~ SPLIT INPUT AND OUTPUT VARIABLES ~~~~~~~~~~~~~~~~~~~~

X = data_for_model.drop("customer_loyalty_score", axis = 1)
y = data_for_model["customer_loyalty_score"]

# ~~~~~~~~~~~~~~~~~~~~~~ SPLIT OUT TRAINING AND TEST SETS ~~~~~~~~~~~~~~~~~~~~

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# ~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH CATEGORICAL VARIABLES ~~~~~~~~~~~~~~~~~~~~~

# Random Forest need input in numeric values so we will keep this section.

# We have a categorical variable gender. Dealing with it using the code from One_Hot_Encoding - 
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse = False, drop = "first") 

# encoder_vars_array = one_hot_encoder.fit_transform(X[categorical_vars])
# Rather than doing the above, we would run the fit method on only the train data.
# This is done because we want the model to learn from the train data and apply it to teh test data.
# This ensures rules will always be the same.
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars]) # Note - this is only transform and not fit_transform.

encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE SELECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In a random forest each variable is judged indepenently. 
# So we don't necessarily need to do feature selection. 
# Although doing feature selection won't have a negative effect on the model accuracy.
# It can infact help with computational efficiency in case the number of predictors is huge.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

regressor = RandomForestRegressor(random_state = 42)
regressor.fit(X_train, y_train)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL ASSESSMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# PREDICT ON THE TEST SET

y_pred = regressor.predict(X_test)

# CALCULATE R-SQUARED

r_squared = r2_score(y_test, y_pred)
print(r_squared)

# CROSS VALIDATION

cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

# CALCULATE ADJUSTED R-SQUARED

num_data_points, num_input_vars = X_test.shape # Returns number of rows and columns
adjusted_r_squared = 1 - (1-r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE IMPORTANCE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This method uses the method of MSE to calculate the feature importance.
# It's not completely reliable as it inflates the importance for numeric variables.

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Using permutation importance - 
# It randomizes one of the variable and checks the model performance to judge the importance of the variable.
# This is many times preferred compared to the first method.

# We can specify how many times we want to apply the reshullfing to a particular variable using n_repeat
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"], permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~ PREDICTIONS UNDER THE HOOD ~~~~~~~~~~~~~~~~~~~~~~~~

y_pred[0]

# We now try analyze the prediction given by each separate decision tree in the random forest and then understand how we got the prediction above.
new_data = [X_test.iloc[0]]
regressor.estimators_

predictions = []
tree_count = 0

for tree in regressor.estimators_ :
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1

print(predictions)
sum(predictions) / tree_count
# This is exactly the same as we got above.

# ~~~~~~~~~~~ SAVING THE MODEL AND REQUIRED OBJECTS FOR PREDICTION ~~~~~~~~~~~

# We save the model so that we can use it to make predition in the future.
# We also have to save any object which may help us in preparing the data just like we did for training dataset.
# So in this case we will require the hot encoding object used to convert the gender categorical variable.

pickle.dump(regressor, open("data/random_forest_regression_model.p", "wb"))
pickle.dump(one_hot_encoder, open("data/random_forest_regression_ohe.p", "wb"))


