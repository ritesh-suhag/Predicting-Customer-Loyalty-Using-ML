
##############################################################################
# LINEAR REGRESSION BASIC TEMPELATE
##############################################################################

# IMPORT REQUIRED PACKAGES 

from sklearn.linear_model import LinearRegression
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

regressor = LinearRegression()

# TRAIN OUR MODEL

regressor.fit(X_train, y_train)

# ASSESS MODEL ACCURACY

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

