import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Fix seed of random number 
np.random.seed(0)

# Create simulated data for 100 people using a 2D Gaussian distribution
mean = [10,10]
cov = [[10,3], [3,10]]
x1, y1 = np.random.multivariate_normal(mean, cov, 100).T
true_false = np.random.rand(100) > 0.9
label1 = np.where(true_false, 1, 0)

# Create simulated data for 100 people using a 2D Gaussian distribution
mean = [20,20]
cov = [[8,4], [4,8]]
x2, y2 = np.random.multivariate_normal(mean, cov, 100).T
true_false = np.random.rand(100) > 0.1
label2 = np.where(true_false, 1, 0)

# Draw data
X = (np.r_[x1, x2])
Y = (np.r_[y1, y2])
label = (np.r_[label1, label2])

# plt.scatter(X[label == 1], Y[label == 1], marker='^', s=30, c='blue', edgecolors='', label='1:continue')
# plt.scatter(X[label == 0], Y[label == 0], marker=',', s=30, c='red', edgecolors='', label='0:withdraw')
plt.xlabel("Annual number of purchases")
plt.ylabel("Average purchase price")
plt.legend(loc="lower right")

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

# Split into training and test data
Data = (np.c_[X, Y])
X_train, X_test, y_train, y_test = train_test_split(Data, label, random_state=0)

# Apply logistic regression
clf = linear_model.LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Plot learned discriminant plane and test data
plot_decision_regions(X_test, y_test, clf=clf, res=0.01, legend=2)

# Evaluate performance with test data
print(f'Accuracy: {clf.score(X_test, y_test):.2f}')

# Classify unknown data
label_prenew = clf.predict([[20, 15]]) # the value [20, 15] is appropriate
print(f'The label of the new customer is {label_prenew}.')

plt.show()