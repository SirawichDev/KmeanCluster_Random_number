# -*- coding: utf-8 -*-

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans



LOADER = np.loadtxt('float_gen.txt')

results=[]
values = np.arange(2,20)


for num_clusters in values:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)
    kmeans.fit(LOADER)
    score = metrics.silhouette_score(LOADER, kmeans.labels_, metric='euclidean', sample_size=len(LOADER))
    results.append(score)
    


plt.bar(values, results, width=.5, color='green', align='center')
plt.title('Silhoustte score Algoritm')


num_cluesters = np.argmax(results) + values[0]
plt.figure()

plt.show()