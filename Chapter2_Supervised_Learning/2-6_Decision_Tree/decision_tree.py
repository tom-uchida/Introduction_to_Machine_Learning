from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Prepare datasets
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=41)

# Apply Dicision Tree
tree = DecisionTreeClassifier(max_depth=None, criterion='gini', random_state=41)
tree.fit(X_train, y_train)

# Print accuracy
print(f'Train Accuracy: {tree.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {tree.score(X_test, y_test):.3f}')