import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from sklearn.datasets import make_blobs

# Prepare dataset
X, y = make_blobs(n_samples=1500, n_features=2, centers=2, random_state=2)

# Expand feature scale
X[:, 0] = X[:, 0] * 10

# Visualize data distribution tendency
# plt.scatter(X[:, 0], X[:, 1])
# plt.title("Blobs Data")
# plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Normalize data
X_norm = MinMaxScaler().fit_transform(X)

# Apply k-means
kmeans = KMeans(n_clusters=2, random_state=0)
y_pred = kmeans.fit_predict(X_norm)

# Visualize the result of k-means clustering
# plt.figure(figsize=(10,5))
# plt.subplot(121)
# plt.scatter(X_norm[:, 0], X_norm[:, 1], c=~y_pred, cmap='bwr')
# plt.title("(a) k-means cluster")
# plt.subplot(122)
# plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap='bwr')
# plt.title("(b) true cluster")
# plt.show()

# Apply k-means
kmeans = KMeans(n_clusters=2, random_state=0)
y_pred = kmeans.fit_predict(X)

# Visualize the result of k-means clustering
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=~y_pred, cmap='bwr')
plt.title("(a) k-means cluster")
plt.subplot(122)
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap='bwr')
plt.title("(b) true cluster")
plt.show()