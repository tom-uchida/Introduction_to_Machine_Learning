import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

plt.style.use('seaborn-darkgrid')

# Generare data
X = np.zeros((20, 2))
X[0:10, 1] = range(0, 10)
X[10:20, 1] = range(0, 10)
X[0, 0] = 1.0
X[9, 0] = 1.0
X[1:9, 0] = 3.0
X[10:20, 0] = range(-1, -11, -1)
X[9, 0] = 1
X[19, 0] = -1

y = np.zeros((20))
y[10:20] = 1.0
y = y.astype(np.int8)

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

# Standardization of variable x
sc = preprocessing.StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# Apply LinearSVC
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
clf = svm.LinearSVC(random_state=0)
clf.fit(X_std, y)
plot_decision_regions(X_std, y, clf=clf, res=0.01)

plt.show()