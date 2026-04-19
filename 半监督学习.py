"""
逻辑回归优化方法
1.选取 离簇最近的特征点 进行训练
2.选取 离簇最近的 n 个特征点 进行训练
"""
#导包
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x,y = load_digits(return_X_y=True)
# print(x)
# print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=52,shuffle=True)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#基本逻辑回归
def dm01():
    log_reg = LogisticRegression(max_iter=1000)
    num_clusters = 50
    x = x_train[:num_clusters]
    y = y_train[:num_clusters]
    log_reg.fit(x,y)
    print(log_reg.score(x_test,y_test))
#聚类选取50特征点后逻辑回归
def dm02():
    kmeans = KMeans(n_clusters = 50,random_state=34)
    x_examples = x_scaled.shape[0]
    num_clusters = 50
    # 计算每个样本到50个中心点的距离
    X_dist = np.zeros((x_examples, num_clusters))
    X_dist = kmeans.fit_transform(x_scaled)
    labels = kmeans.labels_
    # 获取每个样本到50个中心点的距离最小的索引
    x_present_ids = np.argmin(X_dist, axis=0)
    x_present = x_scaled[x_present_ids]
    y_present = y_train[x_present_ids]
    # print(x_present_ids)
    # print(X_dist)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_present,y_present)
    print(log_reg.score(x_test,y_test))
    return X_dist,labels
def dm03(X_dist, labels):
    num_examples = x_train.shape[0]
    num_clusters = np.unique(labels).shape[0]
    #获取每个样本到其中心点的距离
    x_cluster_dist = X_dist[range(num_examples),labels]
    # print(x_cluster_dist)
    percent = 20
    for i in range(num_clusters):
        #判断在该簇内的样本点
        in_clusters = labels == i
        clusters_dist = x_cluster_dist[in_clusters]
        #获取前百分比所对应的数值
        cutoff_dist = np.percentile(clusters_dist,percent)
        above_dist = x_cluster_dist >= cutoff_dist
        #将大于前percent所对应的数值的样本点置为-1
        x_cluster_dist[above_dist & in_clusters] = -1
    partially_propagated = x_cluster_dist != -1
    x_train_partially = x_scaled[partially_propagated]
    y_train_partially = y_train[partially_propagated]
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_train_partially,y_train_partially)
    print(log_reg.score(x_test,y_test))
if __name__ == "__main__":
    dm01()
    X_dist, labels = dm02()
    dm03(X_dist,labels)