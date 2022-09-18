import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

data = pd.read_excel(r"D:\工作文件\2022下半年（研究生）工作文件\油气人工智能基础及应用\data.xlsx")
X = data.values[:, [0, 1, 2]]
y = data.values[:, -1]

scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=5)
X_skernpca = scikit_kpca.fit_transform(X) # 映射

# 可视化
plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], color='red',  alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], color='blue', alpha=0.5)
plt.scatter(X_skernpca[y == 2, 0], X_skernpca[y == 2, 1], color='black', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
