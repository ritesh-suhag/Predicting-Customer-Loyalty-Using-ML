
##############################################################################
# LINEAR REGRESSION - ABC GROCERY TASK
##############################################################################

# ~~~~~~~~~~~~~~~~~~~ IMPORT REQUIRED PACKAGES ~~~~~~~~~~~~~~~~~~~

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

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

# Dealing with outlier requires level playing field - 
outlier_investigation = data_for_model.describe()

# Based on a rough look from the describe we see potential outliers in - 
# 1. distance from store
# 2. total sales
# 3. total items
# Dealing with the same using box-plot approach from data preparation tutorial -
# We just edit the outlier columns and the name of our data frame.

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns :
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2      # Widening the factor trying not to remove too many outliers.
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[( data_for_model[column] < min_border ) | ( data_for_model[column] > max_border )].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)


# ~~~~~~~~~~~~~~~~~~~~~~ SPLIT INPUT AND OUTPUT VARIABLES ~~~~~~~~~~~~~~~~~~~~

X = data_for_model.drop("customer_loyalty_score", axis = 1)
y = data_for_model["customer_loyalty_score"]

# ~~~~~~~~~~~~~~~~~~~~~~ SPLIT OUT TRAINING AND TEST SETS ~~~~~~~~~~~~~~~~~~~~

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# ~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH CATEGORICAL VARIABLES ~~~~~~~~~~~~~~~~~~~~~

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

regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features is {optimal_feature_count}.")

# To get which variables these are, we update X - 
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.grid_scores_)+1), fit.grid_scores_, marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFECV \n Optimal number of features - {optimal_feature_count}. \n At Score - {round(max(fit.grid_scores_),4)}")
plt.tight_layout()
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

regressor = LinearRegression()
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

# EXTRACT MODEL COEFFICIENTS

coefficients = pd.DataFrame(regressor.coef_)
input_variable_names =  pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stats.columns = ["input_variable", "coefficient"]

# EXTRACT MODEL INTERCEPT

regressor.intercept_



