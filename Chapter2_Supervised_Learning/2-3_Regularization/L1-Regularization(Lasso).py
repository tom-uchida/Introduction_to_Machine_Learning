from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

# Read dataset
X, y = load_extended_boston()

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('seaborn-darkgrid')

import numpy as np

# Apply L1-Regularization(Lasso)
lasso = linear_model.Lasso(alpha=0.01, max_iter=2000, random_state=0)
lasso.fit(X_train, y_train)

# Print weight parameters with non-zero values
print(f'Number of nonzero parameters: {np.count_nonzero(lasso.coef_)}')

# Evaluate the linear model with training and test data
print(f'Train score: {lasso.score(X_train, y_train):.2f}')
print(f'Test score: {lasso.score(X_test, y_test):.2f}')