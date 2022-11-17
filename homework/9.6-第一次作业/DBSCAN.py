import numpy as np
import matplotlib.pyplot as plt
import random


r = 0.2
Minpts = 5
# 处理数据集
data = open(r'iris1.txt')
X, Y, true_labels = [], [], []
coreset = []    # 核心对象集合

for line in data.readlines():
    X.append(float(line.strip().split(',')[0]))
    Y.append(float(line.strip().split(',')[1]))
    true_labels.append(float(line.strip().split(',')[4]))


def distance(point1, point2):   # 计算欧式距离
    return np.sqrt((X[point1] - X[point2]) ** 2 + (Y[point1] - Y[point2]) ** 2)


T = []             # 未访问样本集合
neighbors = []     # 样本的邻域
final = []
for i in range(len(X)):
    T.append(i)
    neighbors.append([])
    for j in range(len(X)):
        dis = distance(i, j)
        if dis <= r:
            neighbors[i].append(j)
    if len(neighbors[i]) >= Minpts:
        coreset.append(i)

while coreset != []:
    T_old = T[:]
    o = random.choice(coreset)
    Q = [o]
    T.remove(o)         # 该核心点已经访问过
    while Q != []:
        q = Q[0]        # 取出该核心点
        Q.remove(Q[0])
        if len(neighbors[q]) >= Minpts:
            D_value = list(set(T) & set(neighbors[q]))
            Q += D_value
            for i in D_value:
                T.remove(i)
    for i in T:
        T_old.remove(i)
    for i in T_old:
        if i in coreset:
            coreset.remove(i)
    final.append(T_old)
print(final)

# 画图
color = ['red', 'blue', 'orange', 'black']
for i in range(len(final)):
    for j in final[i]:
        plt.scatter(X[j], Y[j], c=color[i])
plt.title('DBSCAN')
plt.show()