import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r"data.xlsx")
X = data.values[:, [0, 1, 2]]
y = data.values[:, -1]

# 中心化处理
X = X - X.mean(axis=0)


# 计算样本的协方差矩阵
C = np.cov(X, rowvar=False)

# 求解协方差矩阵的特征值和特征向量
eigen_vals, eigen_vecs = np.linalg.eig(C)
# print(eigen_vals)
# print(eigen_vecs)

W = eigen_vecs[:, 0:2]
T = np.dot(X, W)
print(T[:10])

# 画图
color = np.array(['red', 'green', 'blue'])
plt.scatter(T[:, 0], T[:, 1], c=color[y.astype('int32')])
plt.show()