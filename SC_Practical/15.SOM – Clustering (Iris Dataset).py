import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data
y = iris.target

X_vis = X[:, :2]

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_vis)

plt.figure(figsize=(6,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=clusters)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            marker='X', s=200)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Dataset")
plt.grid()
plt.show()
