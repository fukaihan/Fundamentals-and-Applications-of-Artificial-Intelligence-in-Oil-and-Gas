# 油气人工智能基础
---
# 前言
CUP秋季开的《人工智能基础及应用课程》关于支持向量机SVM进行了课程内容的总结。

# 目录
- SVM的数学原理<br />
- SVM算法简介与分类<br />
- SVM算法python实战<br />
- 参考文献<br />

# SVM的数学原理
支持向量机（Support Vector Machine, SVM）细分的话可以分为三类：<br />
1.硬间隔分类器——hard-margin SVM.<br />
2.软间隔分类器——soft-margin SVM.<br />
3.核分类器——kernel SVM.<br />
在本小节主要介绍最基本的硬间隔SVM的数学原理。首先我们定义间隔margin的定义：<br />

$$
margin(\omega,b)=\underset {x_i}{min} &ensp;  distance(\omega,b,x_i)
$$

其中distance表示样本点 $x_i$ 到超平面 $\omega ^ T x_i+b$ 的距离，则margin表示样本点到分割超平面的**最短距离**。由点到直线的距离公式，上式可以改写为：<br />

$$
margin(\omega,b)=\underset {x_i}{min} &ensp; \frac{1}{||\vec \omega||}|\omega ^ T x_i+b|
$$

由此，我们想构造一个分割超平面，我们想要最大化样本点到分割超平面的最短距离，即最大化margin。由分割超平面的定义，对于二分类问题有：<br />

$$
\begin{cases}
\omega ^ T x_i+b >0,y_i=1 \\
\omega ^ T x_i+b <0, y_i=-1
\end{cases}
$$

由此，我们得到了我们SVM的约束优化问题：<br />

$$
max &ensp; margin(\omega,b)
$$

$$
s.t.\begin{cases}
\omega ^ T x_i+b >0,y_i=1 \\
\omega ^ T x_i+b <0, y_i=-1
\end{cases}
$$

进一步的，将优化问题写做：<br />

$$
\underset{\omega,b}{max}  \underset{x_i}{min}  \frac{1}{||\vec \omega||}|\omega ^ T x_i+b|=\underset{\omega,b}{max} \frac{1}{||\vec \omega||} \underset{x_i,y_i}{min}  y_i|\omega ^ T x_i+b|
$$


$$
s.t. &ensp;  y_i(\omega ^ T x_i+b)>0
$$

对于直线 $\omega ^ T x_i+b$ ,在直线上的点满足  $\omega ^ T x_i+b=0$  ，对于  $\omega ^ T$  和  $b$  如果进行等比例缩放，如  $2\omega ^ T x_i+2b$  仍然表示该超平面。<br />

对于约束条件  $y_i(\omega ^ T x_i+b)>0$  ，一定
  
$$
\exists r>0,s.t. &ensp; \underset{x_i,y_i}{min}  y_i(\omega ^ T x_i+b)=r
$$

其中无论 $r $取什么值，都对 $\omega ^ T$ 和 $b$ 等比例缩放 $r$ 倍，使得 $min y_i((\frac{\omega}{r})^Tx_i+\frac{b}{r})=1$ 。<br />
在优化问题中，我们只是为了求出使得 $\frac{1}{||\vec \omega||}|\omega ^ T x_i+b|$ 最大的 $\omega ^ T$ 和 $b$ ，并不关心优化问题的最大值是多少，为了简化运算，我们使得 $r=1$ ，同时不影响 $\omega ^ T$ 和 $b$ 取值。<br />
这样约束优化问题转化为：

$$
\underset{\omega,b}{max} \frac{1}{||\vec \omega||}
$$


$$
s.t. &ensp;  y_i(\omega ^ T x_i+b) \geq1
$$

为了求解优化问题，将其转化为有 $N$ 个约束的二次优化问题：

$$
\underset{\omega,b}{min} \frac{1}{2}\omega^T \omega
$$


$$
s.t. &ensp;  1 - y_i(\omega ^ T x_i+b) \leq 0
$$

构造拉格朗日函数：

$$
L(\omega^T,b,\lambda)=\frac{1}{2}\omega^T \omega+\sum_{i=1}^{N} \lambda_i( 1 - y_i(\omega ^ T x_i+b))
$$

利用Lagrange函数转化为无约束优化问题：<br />

$$
\underset{\omega,b}{min} \underset{\lambda}{min}L(\omega^T,b,\lambda)
$$


$$
s.t. &ensp; \lambda_i \geq 0
$$

利用对偶优化求解上述无约束优化问题，写出上述问题的对偶问题：

$$
\underset{\lambda}{max} \underset{\omega,b}{min}L(\omega^T,b,\lambda)
$$


$$
s.t. &ensp; \lambda_i \geq 0
$$

对于无约束优化问题求解最优值，很简单的一个办法就是利用导数，首先对于固定的 $\lambda$ ，求解使得 $L$ 最小的 $\omega ^ T$ 和 $b$ ，则分别对它们求偏导：

$$
\frac{\partial L}{\partial b}=\frac{\partial }{\partial b}[\sum_{i=1}^{N} \lambda_i-\sum_{i=1}^{N} \lambda_iy_i(\omega^Tx_i+b)]  \\\
=\frac{\partial }{\partial b}[-\sum_{i=1}^{N}\lambda_iy_yb]
$$

得到：

$$
-\sum_{i=1}^{N}\lambda_iy_i=0
$$

此时的Lagrange函数有：<br />

$$
L(\omega^T,b,\lambda)&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
$$


$$
=\frac{1}{2}\omega^T \omega+\sum_{i=1}^{N} \lambda_i( 1 - y_i(\omega ^ T x_i+b))&ensp;&ensp;&ensp;&ensp;
$$


$$
=\frac{1}{2}\omega^T \omega +\sum_{i=1}^{N} \lambda_i-\sum_{i=1}^{N} \lambda_iy_i(w^Tx_i+b)&ensp;&ensp;
$$


$$
=\frac{1}{2}\omega^T \omega +\sum_{i=1}^{N} \lambda_i-\sum_{i=1}^{N} \lambda_iy_iw^Tx_i &ensp;&ensp;&ensp;&ensp;&ensp;
$$

此时再对 $\omega$ 求偏导：

$$
\frac{\partial L}{\partial \omega}= \frac{1}{2}2\omega-\sum_{i=1}^{N}\lambda_iy_ix_i=0\Rightarrow\omega=\sum_{i=1}^{N}\lambda_iy_ix_i
$$

将 $\omega$ 的值带入后得到 $L$ 的最小值：

$$
minL(\omega^T,b,\lambda)&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
$$


$$
=\frac{1}{2}(\sum_{i=1}^{N}\lambda_iy_ix_i)^T(\sum_{j=1}^{N}\lambda_jy_jx_j)+\sum_{i=1}^{N}\lambda_i-\sum_{i=1}^{N}\lambda_iy_i(\sum_{j=1}^{N}\lambda_jy_jx_j)^Tx_i
$$


$$
=-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jx_ix_j+\sum_{i=1}^{N}\lambda_i &ensp;&ensp;&ensp;&ensp;&ensp;
$$

因此此时的优化问题可以转化为：

$$
\underset{\lambda}{min} &ensp;\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jx_ix_j-\sum_{i=1}^{N}\lambda_i
$$


$$
s.t. &ensp; \lambda_i \geq 0, \sum_{i=1}^{N}\lambda_iy_i=0
$$

由KKT条件：

$$
\begin{cases}
\frac{\partial L}{\partial \omega}=0,\frac{\partial L}{\partial b}=0 ,\frac{\partial L}{\partial \lambda}=0\\[2ex]
\lambda_i(1-y_i(\omega^Tx_i+b))=0\\[2ex]
\lambda_i \geq 0 \\[2ex]
1-y_i(\omega^Tx_i+b) \leq 0
\end{cases}
$$

最后得到上述优化问题的最优解：

$$
\begin{cases}
\omega^*=\sum_{i=0}^{N}\lambda_iy_ix_i\\[2ex]
b^*=y_k-\sum_{i=0}^{N}\lambda_iy_ix_i^Tx_k
\end{cases}
$$

所以最后得到的硬间隔分类器的分割超平面为：

$$
(w^*)^Tx+b^*
$$

## SVM算法介绍与分类
SVM(support vector machine, 支持向量机)是一种**有监督**(supervised learning)的**分类算法**，于1964年提出，是一个非常经典的**机器学习**算法。<br />
人们常说：“SVM有三宝——间隔(margin)、对偶(dual)、核技巧(kernel trick)”，因此根据这三个特性，通常将SVM分为以下三种类别：<br />
- hard-margin SVM硬间隔分类器(最大间隔分类器)
- soft-margin SVM软间隔分类器
- kernel SVM核技巧分类器

对于数据集Data={ $(x_i,y_i)_{i=1}^N$,$x_i\in R^P$, $y_i \in (-1,1)$ }，我们想找到一个划分超平面 $\omega^T+b=0 ，使得该划分超平面对样本局部的“扰动性”最好，即鲁棒性最好，如下图所示：<br />

![找到最好的分类超平面](D:/工作文件/2022下半年（研究生）工作文件/油气人工智能基础及应用/支持向量机/picture4.webp)

<br />不同于神经网络“黑匣子”的特性，SVM在如何求解最优的分类超平面是有严格的数学证明的，利用对偶问题(dual problem)以及Lagrange乘数法，KKT条件等优化方法求解出最优超平面的 $\omega$ 和  $b$ .<br />
## SVM算法python实战
首先是一些库和包的引用：<br />
```python
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

# 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False
```
其中warnings是不显示一些奇怪的警告(warning)，因为题主是在运行了很多遍发现没有问题后才导入的这个库，看着美观一些，所以是否要加这个看大家自己。<br />
接下来的模块是**数据的预处理**，为了突出不同处理方式，在此分别采用原始数据、数据的归一化以及数据的标准化。<br />
其中数据的归一化是采用如下公式：

$$
x=\frac{x-max}{max-min}
$$

进而将数据最终都集中映射到[0,1]之间。对于数据的标准化，采用的公式是：

$$
x=\frac{x-\mu}{\sigma}
$$

进而对数据进行了压缩大小处理，同时还让数据具有特殊特征(平均值为0标准差为1)。python代码如下：
```python
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
```
接下来是构造SVM分类器，采用的是sklearn库中的SVM方法，对数据进行训练(部分代码)：
```python
classifier = svm.SVC(C=1.5, kernel='linear')
classifier.fit(X, Y.ravel())
```
其中，C是惩罚系数，也就是在raft-margin SVM中可以容忍的“犯错”的范围，C越大，代表分类边界容忍分类犯错的程度越大；kernel是核函数，对于线性不可分的数据，我们通常先采用核函数将其映射到高维，再进行分类，在这里我们采用的是线性核函数(linear)。SVM常用的方法如下：
```python
# 分类
svm.SVC()         # 所有类型数据，通过控制核函数
svm.LinearSVC()   # 只针对线性核函数

# 回归
svm.SVR()
svm.LinearSVR()
```
接下来就是结果的呈现，由于代码过长，在此就不贴了，其结果为：<br />
![不同数据处理方法的结果](D:/工作文件/2022下半年（研究生）工作文件/油气人工智能基础及应用/支持向量机/picture1.png)
我们可以看到通过样本的归一化之后样本都分布在[0,1]之间，而标准化将数据集的中心放在0的周围。<br />
为了更形象的展示出分类边界(分类器)的分类效果，我采用了plot_decision_regions方法，这是一个别人写好的库，在《Python机器学习(第2版)》中有对于这个函数的详细介绍，是基于网格划分的原理，我正在看还没看完，等后续看完每个章节后会进行记录，在此先用库中写好的函数。
```python
plot_decision_regions(X, Y, clf=classifier, legend=2)
```
分类边界结果呈现：<br />
![绘制分类边界](D:/工作文件/2022下半年（研究生）工作文件/油气人工智能基础及应用/支持向量机/picture2.png)
<br />

在最后，通常我们会对于样本进行划分，分为测试集(test)和训练集(train)。sklearn中train_test_split方法可以帮助我们做到这点，在此我采用的是0.8为训练集，0.2为测试集：<br />
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
```
通过利用训练集的样本对SVM进行训练(利用.fit()方法)，将测试样本输入训练好的SVM中，采用.predict()方法对测试样本进行分类预测：
```python
clf_predict = svm.SVC(C=1.5, kernel='linear')
clf_predict.fit(X_train, Y_train.ravel())
predict = clf_predict.predict(X_test)
```
train_test_split方法中，每次对于样本的划分都是随机的，在此利用循环来进行10次随机的样本划分，来测试SVM分类器的分类效果，将结果化成折线图：<br />
![绘制分类边界](D:/工作文件/2022下半年（研究生）工作文件/油气人工智能基础及应用/支持向量机/picture3.png)
<br />
以上就是对数据的处理以及应用SVM算法对数据集进行分类的全部内容，如果需要源代码的可以在后台私聊我！
## 参考
[1]《机器学习》周志华. 2016<br />
[2] 《Python机器学习(第2版)》Sebastian Raschka. Vahid Mirjalili<br />
[3] 机器学习-白板推导系列(六)-支持向量机SVM. bilibili. shuhuai008
