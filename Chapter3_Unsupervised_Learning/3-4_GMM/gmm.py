import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
plt.style.use('seaborn-darkgrid')

# Prepate sample data
X, y = make_blobs(n_samples=1500, centers=3, random_state=170)

# Apply transform matrix
transformation = [[0.5, -0.6], [-0.3, 0.8]]
X_aniso = np.dot(X, transformation)

# Visualize the result of transformation
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.title("(a) Original")
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.subplot(122)
# plt.title("(b) Anisotropically")
# plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y, cmap='brg')
# plt.show()

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Standardization the data
X_norm = StandardScaler().fit_transform(X_aniso)

# Apply k-means
kmeans = KMeans(n_clusters=3, random_state=5)
kmeans.fit(X_norm)
kmean_y_pred = kmeans.predict(X_norm)

# Apply GMM
gmm = GaussianMixture(n_components=3, random_state=5)
gmm.fit(X_norm)
gmm_y_pred = gmm.predict(X_norm)
idx_label0, idx_label1, idx_label2 = np.where(gmm_y_pred==0), np.where(gmm_y_pred==1), np.where(gmm_y_pred==2)
gmm_y_pred[idx_label0], gmm_y_pred[idx_label1], gmm_y_pred[idx_label2] = 1, 2, 0

# Visualize the result of the clustering
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("(a) true cluster")
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap='brg')
plt.subplot(132)
plt.title("(b) k-means cluster")
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=~kmean_y_pred, cmap='brg')
plt.subplot(133)
plt.title("(c) GMM cluster")
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=gmm_y_pred, cmap='brg')
plt.show()