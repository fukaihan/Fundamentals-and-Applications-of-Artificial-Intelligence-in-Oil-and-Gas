import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r"D:\工作文件\2022下半年（研究生）工作文件\油气人工智能基础及应用\data.xlsx")
X = data.values[:, [1, 2, 3]]
y = data.values[:, -1]

x0, x1, x2 = X[y == 0], X[y == 1], X[y == 2]
# 类内均值
mean0, mean1, mean2 = x0.mean(axis=0), x1.mean(axis=0), x2.mean(axis=0)
# 类间均值
mean = X.mean(axis=0)

# 协方差矩阵
c0, c1, c2 = np.cov(x0, rowvar=False), np.cov(x1, rowvar=False), np.cov(x2, rowvar=False)

# 类内散度矩阵
Sw = c0 + c1 + c2
# 类间散度矩阵
Sb0, Sb1, Sb2 = np.outer(mean0 - mean, mean0 - mean), np.outer(mean1 - mean, mean1 - mean), np.outer(mean2 - mean, mean2
                                                                                                     - mean)
Sb = Sb0 + Sb1 + Sb2

# 求Sb逆矩阵
Sb_inv = np.linalg.inv(Sb)
# 求特征值与特征向量
A = np.dot(Sb_inv, Sw)
eigen_vals, eigen_vecs = np.linalg.eig(A)
# print(eigen_vals)
# print(len(eigen_vecs))

W = eigen_vecs[:, 0:2]
T = np.dot(X, W)
# print(T[:10])
# 画图
color = np.array(['red', 'green', 'blue'])
plt.scatter(T[:, 0], T[:, 1], c=color[y.astype('int32')])
plt.show()
