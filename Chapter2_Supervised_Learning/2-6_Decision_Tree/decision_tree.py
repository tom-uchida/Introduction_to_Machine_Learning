from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Prepare datasets
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=41)

# Apply Dicision Tree
tree = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=41)
tree.fit(X_train, y_train)

# Print accuracy
print(f'Train Accuracy: {tree.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {tree.score(X_test, y_test):.3f}')

import graphviz
from sklearn.tree import export_graphviz

# # Export the result of decision tree
# dot_data = export_graphviz(tree, out_file=None, impurity=False, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)

# # Show 
# graph = graphviz.Source(dot_data)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Visualize the importance of each feature
n_features = wine.data.shape[1]
plt.title('Feature Importances')
plt.bar(range(n_features), tree.feature_importances_, align='center')
plt.xticks(range(n_features), wine.feature_names, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()