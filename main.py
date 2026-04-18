"""
K-meand算法原理：将数据(data)划分为 n 个簇,首先随机在数据(data)中选取 n 个点作为中心点,
算取数据(data)其他点到中心点的距离,若该数据点到 某个中心点 的距离最小，则该数据点划分为此中心点内,
算完所有数据点(data)后,更新中心点(更新方法为：计算该中心点内的所有数据点的平均值作为新的中心点),
重复上述步骤,直到各个中心点内的数据点(data)变化趋于稳定,划分结束
"""

#导包
import numpy as np
#定义K-means类
class Kmeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters
    def train(self, iterations):
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples,1))
        #调用自定义初始化中心点方法(自定义)
        centroids = Kmeans.init_centroids(self.data, self.num_clusters, num_examples)
        #迭代
        for _ in range(iterations):
            #寻调用寻找最接近中心点的方法(自定义)
            closest_centroids_ids = Kmeans.centroids_find_closest(self.data, centroids)
            centroids = Kmeans.centroids_update(self.data, self.num_clusters, closest_centroids_ids)
        return centroids, closest_centroids_ids
    #@staticmethod修饰静态方法,修饰类方法后,不会自动传入self参数,调用静态方法时,需要加上类名.方法名,静态方法不能实例化对象调用
    @staticmethod
    #定义初始化中心点方法
    def init_centroids(data,num_clusters,num_examples):
        #np.random.permutation(n)用于生成0到n-1的随机数列
        centroids_ids = np.random.permutation(num_examples)
        #取centroids_ids的前num_clusters个数作为中心点
        centroids = data[centroids_ids[:num_clusters],:]
        return centroids
    @staticmethod
    #定义寻找最接近中心点的方法
    def centroids_find_closest(data, centroids):
        #获取中心点的个数
        centroids_len = centroids.shape[0]
        #获取数据点的个数
        examples_len = data.shape[0]
        #初始化最接近中心点为0
        closest_centroids_ids = np.zeros((examples_len,1))
        for i in range(examples_len):
            closest_centroids_ids[i] = np.argmin([np.sum((data[i]-centroid)**2) for centroid in centroids])
        return closest_centroids_ids
    #定义中心点更新方法
    @staticmethod
    def centroids_update(data, num_centroids, closest_centroids_ids):
        #获取数据的特征个数，列数
        centroids_features = data.shape[1]
        #初始化中心点
        centroids = np.zeros((num_centroids,centroids_features))
        for centroids_ids in range(num_centroids):
            clusters_mask = closest_centroids_ids.flatten() == centroids_ids
            #防止中心点为空报错
            if clusters_mask.sum() > 0:
                centroids[centroids_ids] = np.mean(data[clusters_mask],axis = 0)
        return centroids
