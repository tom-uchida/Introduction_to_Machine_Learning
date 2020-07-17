from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Prepare datasets
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=41)

# Apply Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=7, max_features=3, max_depth=3, criterion='gini', random_state=41)
forest.fit(X_train, y_train)

# Print accuracy
print(f'Train Accuracy: {forest.score(X_train, y_train):.3f}')
print(f'Test Accuracy: {forest.score(X_test, y_test):.3f}')

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

n_features = wine.data.shape[1]
plt.title('Feature Importances')
plt.bar(range(n_features), forest.feature_importances_, align='center')
plt.xticks(range(n_features), wine.feature_names, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()