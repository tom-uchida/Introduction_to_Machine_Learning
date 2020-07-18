import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

np.random.seed(0)

# Generate pseudo data for 300 people with using the 2D normal distribution
mean = [165, 60]
cov = [[15, 20], [20, 10]]
X = np.random.multivariate_normal(mean, cov, 300)

# Draw figure
# plt.figure(figsize=(6,6))
# plt.scatter(X.T[0], X.T[1], marker='^')
# plt.xlim(150, 180)
# plt.ylim(45, 75)
# plt.xlabel("height")
# plt.ylabel("weight")
# plt.show()

from sklearn import preprocessing, decomposition

# Normalize the data
sc = preprocessing.StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# Do PCA
pca = decomposition.PCA(random_state=0)
pca.fit(X_std) # Calculation of PCA
X_pca = pca.transform((X_std)) # Apply PCA to X_std

# Show the result of PCA
print("主成分の分散説明率(寄与率)")
print(pca.explained_variance_ratio_)

print("固有ベクトル")
print(pca.components_)

pca_point1 = sc.mean_ - sc.inverse_transform(pca.components_)[0]
pca_point2 = sc.mean_ + sc.inverse_transform(pca.components_)[0]
pca_point = np.c_[pca_point1, pca_point2]

# Draw the data
plt.figure(figsize=(6,6))
plt.scatter(X.T[0], X.T[1], marker='^')
plt.plot(pca_point[0], pca_point[1], color='black', label='first_PC_axis')
plt.xlim(150, 180)
plt.ylim(45, 75)
plt.xlabel("height")
plt.ylabel("weight")
plt.legend()
plt.show()