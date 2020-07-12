from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

# Read dataset
X, y = load_extended_boston()

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('seaborn-darkgrid')

# Apply multiple regression model
lm_multiple = linear_model.LinearRegression()
lm_multiple.fit(X_train, y_train)

# Print bias and weight parameters
print(f'intercept: {lm_multiple.intercept_:.2f}')
print(f'coef: {lm_multiple.coef_[:4]}')

# Evaluate single linear model with training and test data
print(f'Train score: {lm_multiple.score(X_train, y_train):.2f}')
print(f'Test score: {lm_multiple.score(X_test, y_test):.2f}')

# Visualize training data and predicted single regression model
# plt.xlabel('RM')
# plt.ylabel('MEDV')
# plt.scatter(X_train_single, y_train)
# plt.plot(X_train_single, y_pred_train, color='red', linewidth=2)
# plt.show()