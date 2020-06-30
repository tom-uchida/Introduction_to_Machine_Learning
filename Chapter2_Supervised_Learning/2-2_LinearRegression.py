from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

# Read dataset
X, y = load_extended_boston()

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('seaborn-darkgrid')

# Only use column data for average number of rooms in a house
X_train_single = X_train[:, 5].reshape(-1, 1)
X_test_single = X_test[:, 5].reshape(-1, 1)

# Apply single regression model
lm_single = linear_model.LinearRegression()
lm_single.fit(X_train_single, y_train)

# Predict training data with the constructed single linear model
y_pred_train = lm_single.predict(X_train_single)

# Print bias and weight parameters
print(f'intercept: {lm_single.intercept_:.2f}')
print(f'coef: {lm_single.coef_[0]:.2f}')

# Evaluate single linear model with training and test data
print(f'Train score: {lm_single.score(X_train_single, y_train):.2f}')
print(f'Test score: {lm_single.score(X_test_single, y_test):.2f}')

# Visualize training data and predicted single regression model
# plt.xlabel('RM')
# plt.ylabel('MEDV')
# plt.scatter(X_train_single, y_train)
# plt.plot(X_train_single, y_pred_train, color='red', linewidth=2)
# plt.show()