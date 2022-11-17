import numpy as np
import matplotlib.pyplot as plt

# 处理数据集
data = open(r'iris1.txt')
X = []
Y = []
true_labels = []

for line in data.readlines():
    X.append(float(line.strip().split(',')[0]))
    Y.append(float(line.strip().split(',')[1]))
    true_labels.append(int(line.strip().split(',')[4]))


def distance(cluster1, cluster2):  # 平均链接算法求距离
    _distance = []
    for i in cluster1:
        for j in cluster2:
            _distance.append(np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2))
    dis = list(set(_distance))
    return np.mean(dis)


def Agnes(k):        # k是分类数
    labels = []
    for i in range(len(X)):
        labels.append([i])
    while len(labels) > k:
        temp = []
        for i in range(len(labels)):      # 遍历所有点,计算他们的距离
            for j in range(i + 1, len(labels)):
                dis = distance(labels[i], labels[j])
                temp.append([i, j, dis])

        temp = sorted(temp, key=lambda x: x[2])    # 按照距离升序排列
        i, j = temp[0][0], temp[0][1]              # i,j代表距离最近的两个点的序号
        merge = labels[i] + labels[j]              # 将距离最近的两个簇归为一类
        labels = [labels[t] for t in range(len(labels)) if t != i and t != j]
        labels.append(merge)

    # 画图
    color = ['red', 'blue', 'orange', 'black']
    for i in range(len(labels)):
        x = []
        y = []
        for j in range(len(labels[i])):
            x.append(X[labels[i][j]])
            y.append(Y[labels[i][j]])
        plt.scatter(x, y, c=color[i])
    plt.legend(['C1', 'C2', 'C3', 'C4'])
    plt.title('AGNES')
    plt.show()

    predict_lables = []        # 预测的类别
    count = 1
    for i in labels:
        for j in range(len(i)):
            predict_lables.append(count)
        count += 1

    return predict_lables


predict_lables = Agnes(3)
print(predict_lables)