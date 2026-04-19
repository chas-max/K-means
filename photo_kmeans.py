"""本文档演示通过KMeans进行图像分割"""
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#读取图片
data = plt.imread('ladybug.png')
# print(type(data))
# print(data.shape)
#对图片进行降维
X = data.reshape(data.shape[0]*data.shape[1],3)
# print(data.shape)
#创建KMeans模型
kmeans = KMeans(n_clusters=8, n_init=5,init='random',random_state=52).fit(X)
#获取标签
labels = kmeans.labels_ #type:ignore
# print(labels)
#获取簇的中心点
centroids = kmeans.cluster_centers_ #type:ignore
# print(centroids)
#将簇内的数据点进行聚类，转变为簇中心点的值
clusters = centroids[labels]
# print(clusters)
# print(data.shape[0],data.shape[1])
print(clusters.shape)
clusters = clusters.reshape(data.shape[0],data.shape[1],3)
plt.figure(figsize=(10,8))
plt.imshow(clusters)
plt.title('图像分割结果')
plt.show()