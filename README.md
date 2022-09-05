# 油气人工智能基础<br />
---
# 聚类算法
定义：聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个簇(cluster).通过这样的划分，每个簇可能对应于一些潜在的类别.<br />
>在聚类文件夹中，采用了根据算法流程图分别实现K-means、Agnes、GMM和DBSCAN算法(算法流程图参考老师的PPT以及周志华的西瓜书).文件夹中的数据集，其中data1是6个点的小数据集，用来实现算法的整体结构.watermelon是“西瓜书”202页**表9.1**中的数据.最后，iris数据集是鸢尾花数据集，老师留的作业是以前两维进行聚类.在下面的具体分类中会展示算法流程图以及聚类结果.
## 划分聚类 — K-means<br />
定义：K-means算法是通过**均值**进行数据点的聚类，而且是以**距离**作为数据集样本相似性的指标，我在编写的时候采用的是欧式距离.由于是随机生成初始聚类点，所以初始点的生成会影响聚类结果.<br />目前对于K-means的主要研究方向有：<br />
- 如何尽量准确的选取**k值**<br />
- 如何更好的生成**初始点**<br />
- 对于不同数据集是否可以更换**距离**的定义？<br />
> **输入**：样本集D={ $x_1$ , $x_2$ ,..., $x_n$ }和聚类簇数k<br />
> **过程**：<br />
> 从D中随机选取k个样本作为初始均值向量{ &w_1& , &w_2& ,..., &w_k&}<br />
> **repeat**<br />
> $\qquad$ 令 $C_i$ = $\emptyset$ (i $\leq$ k) <br />
> $\qquad$ **for** j = 1, 2, ..., m **do** <br />
> $\qquad\qquad$  计算样本 $x_j$ 与均值向量点的距离 $d_ij$ <br />
> $\qquad\qquad$  根据距离最近的均值向量确定 $x_j$ 的簇标记 a <br />
> $\qquad\qquad$  将样本&x_j&放入相应的簇 $C_a$ <br />
> $\qquad$  **end for**
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
## 模型聚类 — GMM(Gaussian Mixture Model)<br />
## 密度聚类 — DBSCAN(Density-Based Spatial Clustering of Applications with Noise)<br />
---
