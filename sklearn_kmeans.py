"""
本代码演示通过sklearn框架中的clusters.KMeans类创建KMeans对象，并获取划分结果及其划分评判
num_clusters簇的个数的判断方法:
1.画出KMeans.inertia_(该函数的作用是返回各个数据点到中心点距离的和,该数值越小则代表划分效果越好)返回值的图像，寻找拐点
2.通过KMeans.score()(计算方法:ai代表簇内不相似度,即簇内的某一点到该簇内其他数据点的距离的平均值.
bi代表簇间不相似度,即某一个簇内的点到距离最小的那个簇的簇内数据点的平均值)函数观察最接近 1 的那个点
                score = (bi-ai)/max(ai,bi)
"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

x_axis = 'petal_length'
y_axis = 'petal_width'
data = pd.read_csv('iris.csv')
X = data[[x_axis, y_axis]].values
# print(X,X.dtype)
#创建由1到9个簇的KMeans模型组成的列表
kmeans_per_k = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X) for k in range(1,10)]
inertias = [model.inertia_ for model in kmeans_per_k]  # type: ignore
#score()函数返回的值是负的，因为score()函数返回的是负的轮廓系数，轮廓系数越接近 1 ，则划分效果越好，越接近 -1 效果越差。
scores = [-model.score(X) for model in kmeans_per_k]  # type: ignore
# plt.figure(figsize=(10,8))
# plt.plot(range(1,10),scores,'bo-')
# # plt.axis[]
# plt.show()
labels = kmeans_per_k[3].labels_ #type:ignore
# print( labels)
num_clusters = 3
colors = ['red', 'blue', 'yellow']
for i in range(num_clusters):
    plt.scatter(data[x_axis][labels==i],data[y_axis][labels==i],color=colors[i])
plt.title('KMeans fit')
plt.legend()
plt.show()