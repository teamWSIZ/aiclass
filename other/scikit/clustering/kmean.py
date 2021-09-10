from itertools import islice, cycle

from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import numpy as np

np.random.seed(2)

n_samples = 10 ** 3

# n features == liczba "wymiarów" próbki
# dataset = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)    # dwa kółka
# dataset = datasets.make_blobs(n_samples=n_samples, n_features=8, random_state=1, cluster_std=1.1)  # populacje
dataset = datasets.make_blobs(n_samples=n_samples, n_features=2, random_state=1, cluster_std=[2.2, 1.2, 1.3])

# X = noisy_circles

plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05, hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

X, y = dataset  # X : array-like of shape (n_samples, n_features)

connectivity = kneighbors_graph(X, n_neighbors=default_base['n_neighbors'], include_self=False)

algo = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward', connectivity=connectivity)
# algo = cluster.MiniBatchKMeans(n_clusters=3)
# algo = optics = cluster.OPTICS(min_samples=default_base['min_samples'],
#                                xi=default_base['xi'],
#                                min_cluster_size=default_base['min_cluster_size'])

algo.fit(X)

if hasattr(algo, 'labels_'):
    y_pred = algo.labels_.astype(int)
else:
    y_pred = algo.predict(X)

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']), int(max(y_pred) + 1))))
colors = np.append(colors, ["#000000"])  # nieprzypisane do cluster-ów

plt.subplot(2, 2, 1)

# print(X[:, 0])  # np; extract x[0]

plt.scatter(X[:, 0], X[:, 1], s=6, color=colors[y_pred])
plt.show()
