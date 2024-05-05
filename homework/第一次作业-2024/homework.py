"""
Created on Mon Mar 18 12:06:45 2024
作业主要内容：
1.对于原始数据、归一化处理后的数据分别做可视化分析
2.针对归一化后的数据进行聚类（聚类方法不限），训练样本及测试样本给出正叛率
3.对归一化后的数据做主成分分析

@author: 张瑜芝
"""
#%% 导入所需要的库
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering  
from matplotlib.font_manager import FontProperties

import seaborn as sns
import matplotlib.pyplot as plt

#%% 导入数据以及数据处理
data = pd.read_csv(r"C:\Users\Fu_ture\Desktop\data.csv")
X = data.iloc[:, 0:7]
y = data.iloc[:, 7]
X = np.array(X.values)
y = np.array(y.values)

# 数据标准化
Stand_data = preprocessing.StandardScaler()
X_s = Stand_data.fit_transform(X)

# 数据归一化
nor = preprocessing.MinMaxScaler()
X_n = nor.fit_transform(X)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12, stratify=y, test_size=0.3)
X_train_n = nor.fit_transform(X_train)
X_test_n = nor.fit_transform(X_test)

#%% 绘制基础数据图
def plot_data(data1, file_name):
    sns.pairplot(data1, hue='Type', diag_kind='kde')
    plt.savefig(file_name)
    plt.show()


# 绘制原始数据图
data_orig = pd.DataFrame(X, columns=data.columns[:-1])
data_stand = pd.DataFrame(X_s, columns=data.columns[:-1])
data_n = pd.DataFrame(X_n, columns=data.columns[:-1])
data1 = pd.DataFrame(X_n[:, 1:4:2], columns=data.columns[1:4:2])

data_n['Type'] = data['Type']
data_orig['Type'] = data['Type']
data_stand['Type'] = data['Type']
data1['Type'] = data['Type']
plot_data(data1, '交会图.png')

#%% 进行聚类（归一化）

# K-Means
estimator1 = KMeans(n_clusters=3, max_iter=300)
estimator1.fit(X_train_n)
estimator11 = KMeans(n_clusters=3, max_iter=300)
estimator11.fit(X_test_n)

# Birch
estimator2 = Birch(n_clusters=3, threshold=0.5)
estimator2.fit(X_train_n)
estimator22 = Birch(n_clusters=3, threshold=0.5)
estimator22.fit(X_test_n)

#%% 最后对比结果
font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=14)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_n[:, 1], X_train_n[:, 3], c='red', label='训练样本', alpha=0.5)
plt.scatter(X_test_n[:, 1], X_test_n[:, 3], c='blue', label='测试样本', alpha=0.5)
plt.title('归一化原始数据', fontproperties=font)
plt.xlabel('AC')
plt.ylabel('CNL')
plt.legend(prop=font)
plt.grid(True)
plt.savefig('归一化原始数据图.png')
plt.show()

# 绘制K均值聚类效果图
plt.figure(figsize=(8, 6))
plt.scatter(X_train_n[:, 1], X_train_n[:, 3], c=estimator1.labels_, cmap='viridis', alpha=0.5)
plt.title('训练样本聚类结果', fontproperties=font)
plt.xlabel('AC')
plt.ylabel('CNL')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('K-Means 训练样本聚类结果.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_test_n[:, 1], X_test_n[:, 3], c=estimator11.labels_, cmap='viridis', alpha=0.5)
plt.title('测试样本聚类结果', fontproperties=font)
plt.xlabel('AC')
plt.ylabel('CNL')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('K-Means 测试样本聚类结果.png')
plt.show()

# 绘制Birch聚类效果图
plt.figure(figsize=(8, 6))
plt.scatter(X_train_n[:, 1], X_train_n[:, 3], c=estimator2.labels_, cmap='viridis', alpha=0.5)
plt.title('训练样本聚类结果', fontproperties=font)
plt.xlabel('AC')
plt.ylabel('CNL')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('BIRCH 训练样本聚类结果.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_test_n[:, 1], X_test_n[:, 3], c=estimator22.labels_, cmap='viridis', alpha=0.5)
plt.title('测试样本聚类结果', fontproperties=font)
plt.xlabel('AC')
plt.ylabel('CNL')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('BIRCH 测试样本聚类结果.png')
plt.show()

# 层次聚类
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
agg_clustering.fit(X_train_n)

# 绘制层次聚类效果图
plt.figure(figsize=(8, 6))
plt.scatter(X_train_n[:, 1], X_train_n[:, 3], c=agg_clustering.labels_, cmap='viridis', alpha=0.5)
plt.title('训练样本层次聚类结果', fontproperties=font)
plt.xlabel('AC')
plt.ylabel('CNL')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('层次聚类 训练样本聚类结果.png')
plt.show()

# 对测试样本进行层次聚类
agg_clustering_test = AgglomerativeClustering(n_clusters=3, linkage='average')
agg_clustering_test.fit(X_test_n)

# 绘制测试样本层次聚类效果图
plt.figure(figsize=(8, 6))
plt.scatter(X_test_n[:, 1], X_test_n[:, 3], c=agg_clustering_test.labels_, cmap='viridis', alpha=0.5)
plt.title('测试样本层次聚类结果', fontproperties=font)
plt.xlabel('AC')
plt.ylabel('CNL')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.savefig('层次聚类 测试样本聚类结果.png')
plt.show()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_n)

# 将PCA结果转换为DataFrame
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Type'] = y

# 绘制主成分分析结果
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Type', data=pca_df, palette='viridis')
plt.title('归一化数据的主成分分析结果', fontproperties=font)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(prop=font)
plt.grid(True)
plt.savefig('PCA结果.png')
plt.show()

