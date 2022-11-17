import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#%% 数据处理
data = pd.read_excel(r"data.xlsx")
x = data.iloc[:, [1, 2]].values
y = data.iloc[:, -1].values

X = np.vstack((x[y == 0], x[y == 1]))
Y = np.hstack((y[y == 0], y[y == 1]))

# 数据标准化
scale = StandardScaler()
X_scale = scale.fit_transform(X)

# 数据归一化
x_scale = np.zeros(X.shape)
for i in range(len(X[:, 0])):
    x_scale[:, 0][i] = (X[:, 0][i] - min(X[:, 0])) / (max(X[:, 0]) - min(X[:, 0]))
    x_scale[:, 1][i] = (X[:, 1][i] - min(X[:, 1])) / (max(X[:, 1]) - min(X[:, 1]))


# 构建svm分类器
classifier = svm.SVC(C=1.5, kernel='linear')
classifier.fit(X, Y.ravel())
classifier1 = svm.SVC(C=1.5, kernel='linear')
classifier1.fit(X_scale, Y.ravel())
classifier2 = svm.SVC(C=1.5, kernel='linear')
classifier2.fit(x_scale, Y.ravel())


#%% 画图
color = np.array(['red', 'green', 'blue'])
plt.figure(1)
plt.subplot(1, 3, 1)
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], c=color[0], label='石油')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], c=color[1], label='天然气')
plt.xlabel('AC')
plt.ylabel('DEN')
plt.title('原始')
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(X_scale[Y == 0][:, 0], X_scale[Y == 0][:, 1], c=color[0], label='石油')
plt.scatter(X_scale[Y == 1][:, 0], X_scale[Y == 1][:, 1], c=color[1], label='天然气')
plt.xlabel('AC')
plt.ylabel('DEN')
plt.title('标准化')
plt.legend()
plt.subplot(1, 3, 3)
plt.scatter(x_scale[Y == 0][:, 0], x_scale[Y == 0][:, 1], c=color[0], label='石油')
plt.scatter(x_scale[Y == 1][:, 0], x_scale[Y == 1][:, 1], c=color[1], label='天然气')
plt.xlabel('AC')
plt.ylabel('DEN')
plt.title('归一化')
plt.legend()
plt.show()


plt.figure(2)
plt.subplot(1, 3, 1)
plot_decision_regions(X, Y, clf=classifier, legend=2)
plt.xlabel('AC')
plt.ylabel('DEN')
plt.title('原始')
plt.subplot(1, 3, 2)
plot_decision_regions(X_scale, Y, clf=classifier1, legend=2)
plt.xlabel('AC')
plt.ylabel('DEN')
plt.title('标准化')
plt.subplot(1, 3, 3)
plot_decision_regions(x_scale, Y, clf=classifier2, legend=2)
plt.xlabel('AC')
plt.ylabel('DEN')
plt.title('归一化')
plt.show()

#%% 划分样本集和训练集
plt.figure(3)
prediction = []
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf_predict = svm.SVC(C=1.5, kernel='linear')
    clf_predict.fit(X_train, Y_train.ravel())
    predict = clf_predict.predict(X_test)
    prediction.append(metrics.accuracy_score(predict, Y_test))

x = range(10)
plt.plot(x, prediction)
plt.xlabel('试验次数')
plt.ylabel('准确率')
plt.title('svm分类准确率')
plt.show()