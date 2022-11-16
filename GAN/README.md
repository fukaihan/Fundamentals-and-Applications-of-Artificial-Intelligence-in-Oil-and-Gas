# 油气人工智能基础<br />
---
# 聚类算法
定义：聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个簇(cluster).通过这样的划分，每个簇可能对应于一些潜在的类别.<br />
>在聚类文件夹中，采用了根据算法流程图分别实现K-means、Agnes、GMM和DBSCAN算法(算法流程图参考老师的PPT以及周志华的西瓜书).文件夹中的数据集，其中watermelon是“西瓜书”202页**表9.1**中的数据.最后，iris数据集是鸢尾花数据集，老师留的作业是以前两维进行聚类.在下面的具体分类中会展示算法流程图以及聚类结果.
## 划分聚类 — K-means<br />
定义：K-means算法是通过**均值**进行数据点的聚类，而且是以**距离**作为数据集样本相似性的指标，我在编写的时候采用的是欧式距离.由于是随机生成初始聚类点，所以初始点的生成会影响聚类结果.<br />目前对于K-means的主要研究方向有：<br />
- 如何尽量准确的选取**k值**<br />
- 如何更好的生成**初始点**<br />
- 对于不同数据集是否可以更换**距离**的定义？<br />
> **输入**：样本集D={ $x_1$ , $x_2$ ,..., $x_n$ }和聚类簇数k<br />
> **过程**：<br />
> 从D中随机选取k个样本作为初始均值向量{ $w_1$ , $w_2$ ,..., $w_k$}<br />
> **repeat**<br />
> $\qquad$ 令 $C_i$ = $\emptyset$ (i $\leq$ k) <br />
> $\qquad$ **for** j = 1, 2, ..., m **do** <br />
> $\qquad\qquad$  计算样本 $x$ 与均值向量点的距离 $d_ij$ <br />
> $\qquad\qquad$  根据距离最近的均值向量确定 $x_j$ 的簇标记 a <br />
> $\qquad\qquad$  将样本&x_j&放入相应的簇 $C_a$ <br />
> $\qquad$  **end for**<br />
> $\qquad$  **for** i = 1, 2, ..., k **do** <br />
> $\qquad\qquad$  计算新的均值向量 $w$ <br />
> $\qquad\qquad$  **if** w!= $w_i$ <br />
> $\qquad\qquad\qquad$  更新均值向量 <br />
> $\qquad\qquad$  **else** <br />
> $\qquad\qquad\qquad$   保持不变 <br />
> $\qquad\qquad$  **end if** <br />
> $\qquad$ **end for** <br />
> **until**当前均值向量不更新<br />
> **输出**：当前聚类划分结果C={ $C_1$ , $C_2$ , $C_3$ }<br />
## 层次聚类 — Agnes(AGglomerative NESting)<br />
定义：它先将数据集的每个样本看作是一个初始簇，然后在算法中运行的每一步找出**距离最近**的两个簇进行合并，该过程持续进行，知道达到预设的聚类簇的个数<br />
算法在进行簇与簇的距离计算有多种方法，在这里我才用的是**平均链接方法**(即计算簇内所有点距离和的平均值)<br />
> **输入**：样本集D={ $x_1$ , $x_2$ ,..., $x_n$ }、聚类簇数k和聚类簇距离函数d<br />
> **过程**：<br />
> **for**j = 1, 2,..., m **do** <br />
> $\qquad$ $C_j$ = { $x_j$ } <br />
> **end for**
> **for** i = 1, 2,..., m **do** <br />
> $\qquad$ **for**j = 1, 2,...,m **do** <br />
> $\qquad\qquad$ $M(i,j)$ =d( $C_i$ , $C_j$ ) <br />
> $\qquad\qquad$ $M(j,i)$ = $M(i,j)$ <br />
> $\qquad$ **end for** <br />
> **end for** <br />
> 设置当前聚类簇个数：q=m <br />
> **while** q > k **do** <br />
> $\qquad$ 找出距离最近的两个簇并合并 <br />
> $\qquad$ 更新原来的簇的距离矩阵以及序号 <br />
> $\qquad$ q = q - 1 <br />
> **end while** <br />
> **输出**：当前聚类划分结果C={ $C_1$ , $C_2$ , $C_3$ }<br />
## 模型聚类 — GMM(Gaussian Mixture Model)<br />
定义：涉及**EM**算法以及**多维高斯分布**，算法原理还没太搞懂.<br />
> **输入**：样本集D={ $x_1$ , $x_2$ ,..., $x_n$ }和高斯混合成分个数k. <br />
> **过程**：<br />
> 初始化高斯混合的模型参数 <br />
> **repeat** <br />
> $\qquad$ **for** j = 1, 2,..., m **do** <br />
> $\qquad\qquad$ 计算 $x$ 由各混合成分生成的后验概率 <br />
> $\qquad$ **end for** <br />
> $\qquad$ **for** i = 1, 2,..., k **do** <br />
> $\qquad\qquad$ 计算新的**均值向量**. <br />
> $\qquad\qquad$ 计算新的**协方差矩阵**. <br />
> $\qquad\qquad$ 计算新的**混合系数**. <br />
> $\qquad$ **end for** <br />
> $\qquad$ 更新模型参数 <br />
> **until**满足停止条件 <br />
> $C_i$ = $\emptyset$ <br />
> **for** j =1, 2,...,m **do** <br />
> $\qquad$ 确定簇种类以及标记，更新簇. <br />
> **end for** <br />
> **输出**：当前聚类划分结果C={ $C_1$ , $C_2$ , $C_3$ }<br />
## 密度聚类 — DBSCAN(Density-Based Spatial Clustering of Applications with Noise)<br />
定义：密度聚类算法从样本的密度角度来考虑样本之间的可连接性，并给予可连接样本不断拓展其聚类簇的大小并最终获得结果，DBSCAN算法不需要提前设定聚类数k，其通过“邻域参数”( $\epsilon$ , $MinPts$ )来刻画样本分布的紧密程度<br />
虽然该算法可以自己依照密度进行划分聚类，对于复杂样本(比如作业里的鸢尾花样本)，其参数是很难确定的，因此DBSCAN算法也是我在四个算法中效果最差的.<br />
> **输入**：样本集D={ $x_1$ , $x_2$ ,..., $x_n$ }和邻域参数( $\epsilon$ , $MinPts$ ) <br />
> **过程**：<br />
> 初始化核心对象集合 $coreset$ <br />
> **for** j = 1, 2,..., m **do** <br />
> $\qquad$ 确定样本 $x$ 的邻域 $N$ ( $x_j$ ) <br />
> $\qquad$ **if**邻域中样本个数 $\geq$ $MinPts$ **then** <br />
> $\qquad\qquad$ 将样本$x_j$加入核心对象集合 $coreset$ <br />
> $\qquad$ **end if** <br />
> **end for** <br />
> 初始化聚类簇数：k=0 <br />
> 初始化未访问样本集合：T=D <br />
> **while** $coreset$ != $\emptyset$ <br />
> $\qquad$ 记录当前为访问样本集合，并随机选取一个核心对象寻找它的**密度可达点**. <br />
> $\qquad$ k = k + 1 ,更新聚类簇集合. <br />
> **end while** <br />
> **输出**：当前聚类划分结果C={ $C_1$ , $C_2$ , $C_3$ }<br />

---
