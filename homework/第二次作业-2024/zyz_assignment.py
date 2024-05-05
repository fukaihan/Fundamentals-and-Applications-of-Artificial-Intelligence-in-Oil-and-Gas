import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pydot
from IPython.display import Image
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn import tree
import graphviz
import pydotplus
from io import StringIO
from sklearn.tree import export_graphviz


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X[:,:5], y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义神经网络模型
mlp = MLPClassifier(max_iter=100)

# 定义参数网格
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

# 网格搜索
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', verbose=1) #, random_state=42
grid_search.fit(X_train_scaled, y_train)

# 输出最优超参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最优超参数训练模型
best_mlp = grid_search.best_estimator_
best_mlp.fit(X_train_scaled, y_train)

# 在训练集和测试集上进行预测
y_train_pred = best_mlp.predict(X_train_scaled)
y_test_pred = best_mlp.predict(X_test_scaled)

# 计算准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)

# 绘制模型结构
import networkx as nx

def plot_neural_network(hidden_layer_sizes, activation, solver, alpha):
    G = nx.DiGraph()

    # Add input layer
    G.add_nodes_from(['Input'], subset=0)

    # Add hidden layers
    for i, layer_size in enumerate(hidden_layer_sizes):
        G.add_nodes_from([f'Hidden Layer {i+1} Node {j+1}' for j in range(layer_size)], subset=i+1)
        if i == 0:
            for j in range(layer_size):
                G.add_edge('Input', f'Hidden Layer {i+1} Node {j+1}', weight=1)  # Input to first hidden layer
        else:
            for k in range(hidden_layer_sizes[i-1]):
                for j in range(layer_size):
                    G.add_edge(f'Hidden Layer {i} Node {k+1}', f'Hidden Layer {i+1} Node {j+1}', weight=1)

    # Add output layer
    G.add_nodes_from(['Output'], subset=len(hidden_layer_sizes)+1)
    for k in range(hidden_layer_sizes[-1]):
        G.add_edge(f'Hidden Layer {len(hidden_layer_sizes)} Node {k+1}', 'Output', weight=1)  # Last hidden layer to output

    # Plotting
    pos = nx.multipartite_layout(G, subset_key='subset')
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', alpha=0.8, font_weight='bold')

    # Add information about hyperparameters
    plt.text(1, 1, f'Activation: {activation}\nSolver: {solver}\nAlpha: {alpha}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.title('Neural Network Structure')
    plt.savefig('Neural Network Structure.png',dpi=300)
    plt.show()

# 获取最优超参数信息
best_params = grid_search.best_params_
hidden_layer_sizes = [10,]
activation = best_params['activation']
solver = best_params['solver']
alpha = best_params['alpha']

# 绘制神经网络结构图
plot_neural_network(hidden_layer_sizes, activation, solver, alpha)


# 绘制混淆矩阵图
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(iris.target_names)), iris.target_names)
    plt.yticks(np.arange(len(iris.target_names)), iris.target_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # 在图上添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    # 保存图像到指定文件名
    plt.savefig(filename, dpi=300)
    plt.show()


# 保存训练集混淆矩阵图
plot_confusion_matrix(y_train, y_train_pred,
                      title='Confusion Matrix - Training Set (Accuracy: {:.2f})'.format(train_accuracy),
                      filename='confusion_matrix_train.png')

# 保存测试集混淆矩阵图
plot_confusion_matrix(y_test, y_test_pred, title='Confusion Matrix - Test Set (Accuracy: {:.2f})'.format(test_accuracy),
                      filename='confusion_matrix_test.png')

import matplotlib.pyplot as plt

def plot_model(mlp, file_name='model_plot.png'):
    hidden_layers = mlp.hidden_layer_sizes
    activation = mlp.activation
    solver = mlp.solver
    alpha = mlp.alpha

    num_layers = len(hidden_layers) + 2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title('Neural Network Model\n\n')

    # Draw input layer
    ax.add_patch(plt.Rectangle((0, 0.5), 1, 0.25, fill=True, color='lightblue'))
    ax.text(0.5, 0.625, f'Input Layer\n({X_train.shape[1]} features)', ha='center', va='center')

    # Draw hidden layers
    prev_x = 0
    for i, layer_size in enumerate(hidden_layers):
        ax.add_patch(plt.Rectangle((i + 1, 0), 1, 1, fill=True, color='lightblue'))
        ax.text(i + 1.5, 0.5, f'Hidden Layer {i + 1}\n({layer_size} neurons)', ha='center', va='center')
        if i > 0:
            ax.plot([prev_x + 0.5, i + 0.5], [0.75, 0.25], color='gray', linestyle='-', linewidth=1)
        prev_x = i

    # Draw output layer
    ax.add_patch(plt.Rectangle((num_layers - 1, 0.5), 1, 0.25, fill=True, color='lightblue'))
    ax.text(num_layers - 0.5, 0.625, f'Output Layer\n({len(np.unique(y_train))} classes)', ha='center', va='center')

    ax.set_xlim(0, num_layers)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


plot_model(best_mlp)

#绘制模型结构


# 使用PCA将数据降维至二维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

from sklearn.neural_network import MLPClassifier
#使用pca后的训练集训练mlp模型，再用mlp模型对训练集和训练集进行绘制二维分类边界
mlp = MLPClassifier(hidden_layer_sizes=(4,),random_state=1)
mlp.fit(X_train_pca,y_train)

# 绘制原始数据的散点图
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=40, marker='o', alpha=0.7)
plt.title('Original Data')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.savefig('PCA-原始数据.png')
plt.show()

# 绘制训练集和测试集的散点图，标签使用红色三角形、蓝色圆圈、紫色正方形表示，并在标题中显示准确率
def plot_scatter_with_shapes(X, y, title, accuracy):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    markers = ['^', 'o', 's']
    colors_fill = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
    colors_scatter = ['#FF0000', '#0000FF', '#800080']

    for i in range(3):
        plt.scatter(X[y == i][:, 0], X[y == i][:, 1], c=colors_scatter[i], cmap=cmap_bold, marker=markers[i],
                    label=f'Class {i}', edgecolor='k', s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(f'{title} - Accuracy: {accuracy:.2f}')
    plt.legend()
    plt.savefig(f'{title}.png',dpi=300)
    plt.show()


plot_scatter_with_shapes(X_train_pca, y_train, 'Training Set', train_accuracy)
plot_scatter_with_shapes(X_test_pca, y_test, 'Test Set', test_accuracy)

