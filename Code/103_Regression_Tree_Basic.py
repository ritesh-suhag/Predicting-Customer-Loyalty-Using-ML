
##############################################################################
# REGRESSION TREE - BASIC TEMPLATE
##############################################################################

# IMPORT REQUIRED PACKAGES 

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# IMPORT SAMPLE DATA

my_df = pd.read_csv("data/sample_data_regression.csv")

# SPLIT DATA INTO INPUT AND OUTPUT OBJECTS

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# SPLIT DATA INTO TRAINING AND TEST SETS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# INSTANTIATE OUR MODEL OBJECT

regressor = DecisionTreeRegressor()

# TRAIN OUR MODEL

regressor.fit(X_train, y_train)

# ASSESS MODEL ACCURACY

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

# DEMONSTRATION OF OVER-FITTING

# Decision trees are very prone to over-fitting.

y_pred_train = regressor.predict(X_train)
r2_score(y_train, y_pred_train)
# We see that r2 value for training data is 1 compared to 0.43 on test data.
# This is a clear indication of over-fitting.

# PLOTTING DECISION TREE - 

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)
# We see that the tree has made one mode for each row.

# To limit the decision tree to a particular number we use the min_samples_leaf parameter while instantiating the object.
regressor = DecisionTreeRegressor(min_samples_leaf = 7)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

y_pred_train = regressor.predict(X_train)
r2_score(y_train, y_pred_train)
# Here r2 for train is 0.8 (approx.) and for test is 0.65. Much better than previous result.
# It is still overfitting but lesser than previous example.

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)



