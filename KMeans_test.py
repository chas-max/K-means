import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KMeans_train import Kmeans

#导入数据
data = pd.read_csv('iris.csv')
# print(data.head())
x = 'petal_length'
y = 'petal_width'
y_labels = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
colors = ['red', 'blue', 'yellow']
#绘制真实数据划分及其未划分数据
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
num_y = len(y_labels)
for i in range(num_y):
    plt.scatter(data[x][data['class']== y_labels[i]],data[y][data['class']==y_labels[i]],color=colors[i])
    plt.title('kown data')
plt.subplot(1,2,2)
plt.scatter(data[x], data[y])
plt.title('unknown data')
plt.show()
num_cluters = 3
iteration = 50
#查看数据的形状
# print(f"数据:{data[[x, y]].values},形状:{data[[x, y]].values.shape}, 类型:{type(data[[x, y]].values)}")
#实例化K-means类,data[x,y].values的类型为numpy.ndarray
kmeans = Kmeans(data[[x,y]].values.reshape(data.shape[0],2),3)
#进行训练，得到中心点及其划分
centroids, closest_centroids_ids = kmeans.train(iteration)
#绘制训练的划分结果
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
for i in range(num_y):
    #矩阵可支持布尔索引
    plt.scatter(data[x][data['class']== y_labels[i]],data[y][data['class']==y_labels[i]],color=colors[i])
    plt.title('kown data')
plt.subplot(1,2,2)
for i in range(num_cluters):
    centroids_ids = closest_centroids_ids.flatten() == i
    plt.scatter(data[x][centroids_ids],data[y][centroids_ids],color=colors[i])
    plt.scatter(centroids[i,0],centroids[i,1],color='black',marker='*')
plt.title('kmean data')
plt.legend()
plt.show()
