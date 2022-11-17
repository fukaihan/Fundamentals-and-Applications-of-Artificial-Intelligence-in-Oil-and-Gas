import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

iris = load_iris()
X = iris.data[:]

gmm = GaussianMixture(n_components=4, covariance_type= 'full')
gmm.fit(X)
y_pred = gmm.predict(X)

plt.scatter(X[:,0], X[:, 1], c=y_pred)
plt.show()