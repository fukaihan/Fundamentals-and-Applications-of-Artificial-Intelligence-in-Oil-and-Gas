# 油气人工智能基础
---
# 目录
- 信息增益<br />
- 决策树算法<br />
- 随机森林算法<br />
- KNN算法<br />
- 参考文献<br />

# 信息增益
决策树（Decison tree）算法是可解释性非常强的一种分类算法。我们基于训练集的**特征**，决策树算法通过学习一系列的约束来推断样本所属于的类别。<br />
利用决策树算法，我们从树的根节点开始，通过 **最大的信息增量（largest Information Gain)** 在每个结点来确定样本特征进行分支。在迭代过程中，我们重复分支操作生成子节点直到叶子节点纯度最高。接下来我们重点介绍一下如何最大化信息增量：<br />

为了在具有最多特征信息的节点进行分支，我们需要定义一个目标函数来通过决策树算法进行优化。在这里我们将信息增益（IG）作为我们的优化目标：

$$
IG(D_P,f)=I(D_P)-\sum_{j=1}^{m}\frac{N_j}{N_P}I(D_j)
$$

其中 $f$ 是分支所利用的特征， $D_P$ 和 $D_j$ 分别代表父节点  $ P$ 和 第 $j$ 个子节点 所代表的数据集。 $I$ 是不纯度（impurity measure）， $N_p$ 是父节点的样本点总数， $N_j$ 是第 $j$ 个子节点的样本总数。<br />

从直观可以看到，信息增益（IG）代表了子节点不纯度的和与父节点的不纯度之间的差异。对于二叉树问题，每个父节点分为两个子节点 $D_{left}$ 和$ D_{right}$ ，则此时的信息增益可以写作：<br />

$$
IG(D_P,f)=I(D_P)-\frac{N_{left}}{N_P}I(D_{left})-\frac{N_{right}}{N_P}I(D_{right})
$$

现在对于不纯度的度量准则大致可以分为三种：基尼不纯度（Gini impurity, $I_G$）、熵（entropy, $I_H$）和分类错误率（classification error, $I_E$）。我们首先来学习熵的定义：<br />

对于所有非空类别$p(i|t) \not = 0$，有：

$$
I_H(t)=-\sum_{i=1}^{c}p(i|t)log_2p(i|t)
$$

其中 $p(i|t)$ 代表对于节点 $t$ 样本属于第 $i$ 类的比例。<br />

通过式子我们可以知道，如果这个节点的所有样本都属于同一类则熵为0；当样本是均匀分布时熵最大。对于二分类问题，当 $p(i=1|t)=1$ 或 $p(i=0|t)=0$时熵为0；当样本点在根节点是是均匀分布即$p(i=1|t)=0.5$ 并且$p(i=0|t)=0.5$是熵最大。这样我们同时给出$Gini$指数和分类错误率的定义：

$$
I_G(t)=\sum_{i=1}^{c}p(i|t)(1-p(i|t))=1-\sum_{i=1}^{c}p(i|t)^2
$$
和
$$
I_E=1-max{p(i|t)}
$$
我们可以用下面这个例子来更直观的了解三种方法的计算方式，对于如下例子：<br />

![例子](D:\付楷涵\汉堡的学习日记\Python机器学习(第2版)\第3章\6.png)

图中，我们从父节点 $D_P$ 包括来自类别1的40个样本和来自类别2的40个样本的数据集开始。通过某种特征的筛选我们有两种分类方式，如果我们利用分类错误率$I_E$ 来计算信息增益的话，我们会得到相同的结果：<br />
对于父节点的不纯度：

$$
I_E(D_P)=1-0.5=0.5
$$
其中若采用A种分类方式有：

$$
A:I_E(D_{left})=1-\frac{3}{4}=0.25
$$

$$
A:I_E(D_{right})=1-\frac{3}{4}=0.25
$$

此时对于A类分类器的信息增益$IG$有：

$$
A:IG_E=0.5-\frac{4}{8}0.25-\frac{4}{8}0.25=0.25
$$

同样的，对于B分类器有：

$$
B:I_E(D_{left})=1-\frac{4}{6}=\frac{1}{3}
$$

$$
B:I_E(D_{right})=1-1=0
$$

$$
B:IG_E=0.5-\frac{6}{8} * \frac{1}{3}-0=0.25
$$
我们可以看到，对于这种情况如果采用分类错误率来衡量节点之间的信息增益是不好的，若我们采用$Gini$指数来计算的话：

对于父节点的不纯度：

$$
I_G(D_P)=1-(0.5^2+0.5^2)=0.5
$$

其中若采用A种分类方式有：

$$
A:I_E(D_{left})=1-((\frac{3}{4})^2+(\frac{1}{4})^2)=\frac{3}{8}=0.375
$$

$$
A:I_E(D_{right})=1-((\frac{1}{4})^2+(\frac{3}{4})^2)=\frac{3}{8}=0.375
$$

此时对于A类分类器的信息增益$IG$有：

$$
A:IG_E=0.5-\frac{4}{8}0.375-\frac{4}{8}0.375=0.125
$$

同样的，对于B分类器有：

$$
B:I_E(D_{left})=1-((\frac{2}{6})^2+(\frac{4}{6})^2)=\frac{4}{9}
$$

$$
B:I_E(D_{right})=1-(1^2+0^2)=0
$$

$$
B:IG_E=0.5-\frac{6}{8} * \frac{4}{9}-0=0.16
$$
我们可以看到，对于$Gini$指数而言，B类分类方式具有更大的信息增益，利用同样的方式我们可以计算出熵的结果，在此就不一一计算了。最后我们可以通过画图的方式来更形象的比较三种准则的区别：<br />

```python
import matplotlib.pyplot as plt
import numpy as np
def gini(p):
    return (p)*(1-(p)) + (1 - p) * (1 - (1-p))

def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure(5)
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                    ['Entropy', 'Entropy (scaled)',
                     'Gini Impurity',
                     'Misclassification Error'],
                    ['-', '-', '--', '-'],
                    ['black', 'lightgray',
                     'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab,
                   linestyle=ls, lw=2, color=c)
    
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()
```
结果如下图所示：<br />

![三种分支准则的对比](D:\付楷涵\汉堡的学习日记\Python机器学习(第2版)\第3章\1.png)
# 决策树算法
决策树算法通过构造复杂的决策边界将特征空间分割为矩形，随着决策树深度的增加，决策树的复杂度就越高。下面是利用sklearn库利用决策树算法来实现对鸢尾花数据集的分类：<br />

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=4,
                              random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plt.figure(6)
plot_decision_regions(X_combined,
                      y_combined, 
                      classifier=tree)
plt.xlabel('petal length [cm]')
plt.xlabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
```
分类结果如下所示：<br/ >

![决策树分类效果](D:\付楷涵\汉堡的学习日记\Python机器学习(第2版)\第3章\2.png)
# 随机森林算法
随机森林算法（Random forests）可以看作是多个决策树算法的集成（ensemble），通过以下步骤来构建RF算法：<br />
1.随机的从训练集中**有放回的**抽取n个样本.<br />
2.**无放回的**随机选择d个特征，便于进行节点的切分.<br />
3.重复1、2步骤.<br />
4.利用**多数票**法则来确定预测样本最终所属于的类别.<br />
利用python实现RF算法：<br />
```python
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1)
forest.fit(X_train, y_train)
plt.figure(7)
plot_decision_regions(X_combined,
                      y_combined, 
                      classifier=forest)
plt.xlabel('petal length [cm]')
plt.xlabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show(
```
分类结果如下：<br />

![随机森林分类效果](D:\付楷涵\汉堡的学习日记\Python机器学习(第2版)\第3章\3.png)
