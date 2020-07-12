# L1-Regularization(Lasso)
# L2-Regularization(Ridge)

from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

# Read dataset
X, y = load_extended_boston()

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('seaborn-darkgrid')

# Apply L2-Regularization(Ridge)
ridge = linear_model.Ridge(alpha=1.0, random_state=0)
ridge.fit(X_train, y_train)

# Print bias and weight parameters
print(f'intercept: {ridge.intercept_:.2f}')
print(f'coef: {ridge.coef_[:4]}')

# Evaluate the linear model with training and test data
print(f'Train score: {ridge.score(X_train, y_train):.2f}')
print(f'Test score: {ridge.score(X_test, y_test):.2f}')