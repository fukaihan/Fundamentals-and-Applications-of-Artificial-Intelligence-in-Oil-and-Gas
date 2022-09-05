import numpy as np
import matplotlib.pyplot as plt
import random

# 处理数据集
data = open(r'D:\工作文件\2022下半年（研究生）工作文件\油气人工智能基础及应用\聚类算法\iris1.txt')
X, Y, true_labels, point = [], [], [], []

for line in data.readlines():
    X.append(float(line.strip().split(',')[0]))
    Y.append(float(line.strip().split(',')[1]))
    true_labels.append(float(line.strip().split(',')[4]))
for i in range(len(X)):
    point.append([])
    point[i].append(X[i])
    point[i].append(Y[i])


def distance(x1, x2, y1, y2):   # 计算欧式距离，由于需要更新迭代中心点，所以按照直接的欧式距离进行计算
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


D = []     # 样本集合
C = []     # 分类样本集合
for i in range(len(X)):
    D.append(i)

k = 3      # 聚类样本数
label = random.sample(D, k)     # 初始化k个均值向量点
means = []
for i in label:
    means.append(point[i])
temmeans = means[:]
for i in range(k):
    C.append([])

while True:
    for i in range(k):          # 初始化簇类
        C[i] = []
    for j in range(len(X)):     # 计算每个点与均值向量点的距离
        dis = []
        for i in range(k):
            dis.append(distance(means[i][0], X[j], means[i][1], Y[j]))
        mindis = dis.index(min(dis))
        C[mindis].append(j)
        # print(dis)
        # print(mindis)
    for q in range(k):
        sumX = 0
        sumY = 0
        for i in C[q]:
            sumX += X[i]
            sumY += Y[i]
        newX = sumX / len(C[q])
        newY = sumY / len(C[q])
        if newX != means[q][0] or newY != means[q][1]:
            means[q][0] = newX
            means[q][1] = newY
    if temmeans == means:
        break

print(C)

# 画图
color = ['red', 'blue', 'orange', 'black']
for i in range(len(C)):
    for j in C[i]:
        plt.scatter(X[j], Y[j], c=color[i])
plt.title('k-means')
plt.show()