# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Load the Boston Housing Dataset
boston = load_boston()

# Convert the dataset to a pandas dataframe
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[boston.feature_names], df['PRICE'], test_size=0.2, random_state=42)

# Set up the Lasso Regression model
lasso = Lasso()

# Set up the hyperparameter grid
param_grid = {'alpha': np.logspace(-4, 0, 50)}

# Set up the GridSearchCV object
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best alpha value and mean squared error
print("Best alpha value: ", grid_search.best_params_['alpha'])
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

