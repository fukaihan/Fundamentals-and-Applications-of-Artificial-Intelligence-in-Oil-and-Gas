import pandas as pd
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=1)

plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='s', label='class one')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='class two')
plt.legend(loc='upper right')
plt.show()

##标准化
sc = StandardScaler()
X = sc.fit_transform(X)


##计算分子M矩阵，方式1；
def rbf_kernel_lda_m(X, gamma=0.01, c=0):
    K_m = []
    c_len = len([i for i in y if i == c])
    for row in X:
        K_one = 0.0
        for c_row in X[y == c]:
            K_one += np.exp(-gamma * (np.sum((row - c_row) ** 2)))
        K_m.append(K_one / c_len)
    return np.array(K_m)


##计算M矩阵，方式2，结果同方式1
def rbf_kernel_lda_m_two(X, gamma=0.01, c=5):
    N = X.shape[0]
    c_len = len([i for i in y if i == c])
    K_two = np.zeros((N, 1))
    for i in range(N):
        K_two[i, :] = np.array(np.sum([np.exp(-gamma * np.sum((X[i] - c_row) ** 2)) for c_row in X[y == c]]))
    return K_two / c_len


##计算N矩阵
def rbf_kernel_lda_n(X, gamma=0.01, c=5):
    N = X.shape[0]
    c_len = len([i for i in y if i == c])
    I = np.eye(c_len)
    I_n = np.eye(N)
    I_c = np.ones((c_len, c_len)) / c_len
    K_one = np.zeros((X.shape[0], c_len))

    for i in range(N):
        K_one[i, :] = np.array([np.exp(-gamma * np.sum((X[i] - c_row) ** 2)) for c_row in X[y == c]])
    K_n = K_one.dot(I - I_c).dot(K_one.T)  ##+ I_n*0.001
    return K_n


##计算新样本点映射后的值；alphas 是其中一个映射向量
def project_x(X_new, X, gamma=0.01, alphas=[]):
    N = X.shape[0]
    X_proj = np.zeros((N, 1))
    for i in range(len(X_new)):
        k = np.exp(-gamma * np.array([np.sum((X_new[i] - row) ** 2) for row in X]))
        X_proj[i, 0] = np.real(k[np.newaxis, :].dot(alphas))  ##不能带虚部
    return X_proj


for g_params in list([80, 100, 500]):  ##14.52
    N = X.shape[0]
    ##求判别式广义特征值和特征向量
    K_m0 = np.zeros((N, 1))
    K_m1 = np.zeros((N, 1))
    K_m0 = rbf_kernel_lda_m(X, g_params, c=0)
    K_m1 = rbf_kernel_lda_m(X, g_params, c=1)
    K_m = (K_m0 - K_m1)[:, np.newaxis].dot((K_m0 - K_m1)[np.newaxis, :])

    K_n = np.zeros((N, N))
    for i in np.unique(y):
        K_n += rbf_kernel_lda_n(X, g_params, c=i)

        ##方式1
    from numpy import linalg

    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(K_n).dot(K_m))
    eigen_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    alphas1 = eigen_pairs[0][1][:, np.newaxis]
    alphas2 = eigen_pairs[1][1][:, np.newaxis]

    ##方式2
    from scipy.linalg import eigh

    eigvals1, eigvecs1 = eigh(np.linalg.inv(K_n).dot(K_m))
    eigen_pairs_one = [(np.abs(eigvals1[i]), eigvecs1[:, i]) for i in range(len(eigvals1))]
    eigen_pairs_two = sorted(eigen_pairs_one, key=lambda k: k[0], reverse=True)
    alphas_one = eigen_pairs_two[0][1][:, np.newaxis]
    alphas_two = eigen_pairs_two[1][1][:, np.newaxis]
    # alphas1 = eigvecs1[-1][:, np.newaxis]
    # alphas1 = eigvecs1[-2][:, np.newaxis]

    ##新样本点
    X_new = np.zeros((N, 2))
    X_new[:, 0][:, np.newaxis] = project_x(X[:, :], X, g_params, alphas1)  # alphas_one,最佳参数gamma=14.52
    X_new[:, 1][:, np.newaxis] = project_x(X[:, :], X, g_params, alphas2)  # alphas_two,最佳参数gamma=14.52

    plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='red', marker='s', label='train one')
    plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='blue', marker='o', label='train two')
    plt.legend(loc='upper right')
    plt.show()

    ##使用LR对样本进行分类
    lr = LogisticRegression(C=1000, random_state=1, penalty='l1')
    lr.fit(X_new, y)
    ##绘制决策边界
    pre.plot_decision_regions(X_new, y, lr, resolution=0.02)
    plt.show()
