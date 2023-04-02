# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Boston Housing Dataset
boston = load_boston()

# Convert the dataset to a pandas dataframe
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[boston.feature_names], df['PRICE'], test_size=0.2, random_state=42)

# Fit a Lasso Regression model to the training data
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Print the coefficients of the model
coef = pd.Series(lasso.coef_, index=boston.feature_names)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
print("Coefficients:")
print(coef)


# Make predictions on the testing set and calculate the mean squared error
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

