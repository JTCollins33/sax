import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from nltk import cluster as clustern
# from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
# import sklearn_extensions as ske
# from fcmeans import FCM
import sklearn.decomposition as decomp
import sklearn.manifold as man
import sklearn.preprocessing as pp
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D


class LayerGroup:

    def __init__(self, layers, figs=None):
        self.layers = layers
        self.group_tsbm = self._gather_tsbm()
        # self.group_wavelets = self._gather_wavelets()
        self.group_dist = self._gather_dist()
        self.kmeans_fit = None
        self.tsne_fit = None
        self.pca_fit = None
        self.figs = figs

    def _gather_tsbm(self):
        tsbm = self.layers[0].tsbm
        for l in self.layers[1:]:
            tsbm = np.concatenate((tsbm, l.tsbm), axis=0)
        return tsbm
    
    def _gather_wavelets(self):
        wavelets = self.layers[0].wavelets
        for l in self.layers[1:]:
            wavelets = np.concatenate((wavelets, l.wavelets), axis=0)
        return wavelets

    def _gather_dist(self):
        dist = self.layers[0].distance
        for l in self.layers[1:]:
            dist = np.concatenate((dist, l.distance), axis=0)
        return dist

    @staticmethod
    def _make_random_reference(length, mins, maxs):
        d = []
        for mini, maxi in zip(mins, maxs):
            d.append(np.random.uniform(mini, maxi, length))
        d = np.array(d)
        return np.rot90(d)

    def gap_statistic(self, sample=1000, refs=3, max_clusters=15, standardize=True):
        gaps = np.zeros((len(range(1, max_clusters + 1)),))
        resultsdf = pd.DataFrame({'cluster_count': [], 'gap': []})
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(self.group_tsbm[:, np.invert(mask)])
        for gap_index, k in enumerate(range(1, max_clusters + 1)):
            print("Trying k = {} | Max k = {}".format(k, max_clusters))
            random_batch = np.random.choice(d.shape[0], size=sample)
            d = d[random_batch]
            min_vals = np.amin(d, axis=0)
            max_vals = np.amax(d, axis=0)
            ref_disps = np.zeros(refs)
            for i in range(refs):
                random_reference = self._make_random_reference(d.shape[0], min_vals, max_vals)
                km = cluster.KMeans(k)
                km.fit(random_reference)
                ref_disp = km.inertia_
                ref_disps[i] = ref_disp
            km = cluster.KMeans(k)
            km.fit(d)
            orig_disp = km.inertia_
            gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
            gaps[gap_index] = gap
            resultsdf = resultsdf.append({'cluster_count': k, 'gap': gap}, ignore_index=True)
        return gaps.argmax() + 1, resultsdf

    def silhouette(self, sample=1000, max_clusters=15, standardize=True):
        silhouettes = np.zeros((len(range(1, max_clusters + 1)),))
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(self.group_tsbm[:, np.invert(mask)])
        for i, k in enumerate(range(2, max_clusters + 1)):
            print("Trying k = {} | Max k = {}".format(k, max_clusters))
            km = cluster.KMeans(n_clusters=k).fit(d)
            silhouettes[i] = metrics.silhouette_score(d, km.labels_, sample_size=sample)
        return silhouettes.argmax() + 1, silhouettes

    def calinksi_harabaz(self, sample=1000, max_clusters=15, standardize=True):
        scores = np.zeros((len(range(1, max_clusters + 1)),))
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(self.group_tsbm[:, np.invert(mask)])
        random_batch = np.random.choice(d.shape[0], size=sample)
        d = d[random_batch]
        for i, k in enumerate(range(2, max_clusters + 1)):
            print("Trying k = {} | Max k = {}".format(k, max_clusters))
            km = cluster.KMeans(n_clusters=k).fit(d)
            scores[i] = metrics.calinski_harabaz_score(d, km.labels_)
        return scores.argmax() + 1, scores

    def kmeans(self, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.KMeans(n_clusters=clusters).fit(d)

    def kmeans_wt(self, clusters=5, standardize=True):
        mask = np.all(self.group_wavelets == 0, axis=0)
        d = self.group_wavelets[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.KMeans(n_clusters=clusters).fit(d)

    def kmeans_pca(self, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.pca_fit = cluster.KMeans(n_clusters=clusters).fit(d)

    def mean_shift(self, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.MeanShift().fit(d)

    def kmeans_seeded(self, centers, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.KMeans(n_clusters=clusters)
        self.kmeans_fit.cluster_centers_ = centers
        labels = self.kmeans_fit.predict(d)
        self.kmeans_fit.labels_ = labels

    def EM(self, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = GaussianMixture(n_components=clusters).fit(d)
        labels = self.kmeans_fit.predict(d)
        self.kmeans_fit.labels_ = labels

    def dbscan(self, eps=0.5, min_samples=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(d)

    def kmedoids(self, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = KMedoids(n_clusters=clusters).fit(d)

    def fuzzy_kmeans(self, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        fcm = FCM(n_clusters=clusters)
        self.kmeans_fit = fcm.fit(d)
        self.kmeans_fit.cluster_centers_ = fcm.centers
        self.kmeans_fit.labels_=fcm.u

    def agglomerative(self, clusters=5, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.AgglomerativeClustering(n_clusters=clusters).fit(d)

    def tsne(self, sample=1000, comps=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
             standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        random_batch = np.random.choice(d.shape[0], size=sample)
        d = d[random_batch]
        self.group_dist = self.group_dist[random_batch]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        tsne_func = man.TSNE(n_components=comps, perplexity=perplexity, early_exaggeration=early_exaggeration,
                             learning_rate=learning_rate)
        self.tsne_fit = tsne_func.fit_transform(d)

    def pca(self, sample=1000, comps=2, standardize=True):
        mask = np.all(self.group_tsbm == 0, axis=0)
        d = self.group_tsbm[:, np.invert(mask)]
        # random_batch = np.random.choice(d.shape[0], size=sample)
        # d = d[random_batch]
        # self.group_dist = self.group_dist[random_batch]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        pca_func = decomp.PCA(n_components=comps)
        self.pca_fit = pca_func.fit_transform(d)

    def plot_tsne(self):
        plt.clf()
        plt.scatter(self.tsne_fit[:, 0], self.tsne_fit[:, 1], c=self.group_dist)
        plt.colorbar()
        plt.show()

    def plot_pca(self, num):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        plt.axes().set_aspect('equal', 'datalim')
        plt.scatter(self.pca_fit[:, 0], self.pca_fit[:, 1], c=self.group_dist[:,0], cmap='rainbow', s=1)
        cbar = plt.colorbar()
        cbar.set_label('Distance from Vector Start (mm)', fontsize='large', fontweight='bold', rotation=270,
                       labelpad=15)
        plt.ylabel('Principal Component 2', fontsize='large', fontweight='bold')
        plt.xlabel('Principal Component 1', fontsize='large', fontweight='bold')
        name = 'pca_' +str(num) + '.png'
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)
        # for l in self.layers:
        #     l.plot_pca(max_clusters=4)

    def plot_gap_statistic(self, gapdf, k):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        plt.plot(gapdf.cluster_count, gapdf.gap, linewidth=3)
        plt.grid(True)
        plt.xlabel('Cluster Count', fontsize='large', fontweight='bold')
        plt.ylabel('Gap Statistic', fontsize='large', fontweight='bold')
        name = 'gap' + '.png'
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)

    def plot_group_tsbm(self, suffix=''):
        cc = self.kmeans_fit.cluster_centers_
        labels = ['aa', 'ab', 'ac', 'ad', 'ae',
                  'ba', 'bb', 'bc', 'bd', 'be',
                  'ca', 'cb', 'cc', 'cd', 'ce',
                  'da', 'db', 'dc', 'dd', 'de',
                  'ea', 'eb', 'ec', 'ed', 'ee']
        labels = np.array(labels)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        mask = np.all(self.group_tsbm == 0, axis=0)
        plt.xticks(range(np.count_nonzero(np.invert(mask))), labels[np.invert(mask)])
        plt.ylabel('Standardized Frequency', fontsize='large', fontweight='bold')
        name = 'group_tsbm' + suffix + '.png'
        color = iter(plt.cm.rainbow(np.linspace(0, 1, cc.shape[0])))
        for i in range(cc.shape[0]):
            c = next(color)
            label = 'Cluster ' + str(i)
            plt.plot(cc[[i], :][0], label=label, linewidth=3, marker='o', linestyle='--', c=c)
        ax.legend()
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)

    def plot_kmeans(self, max_clusters=None):
        ind = 0
        for l in self.layers:
            dist = len(l.tsbm)
            start = ind
            end = ind + dist
            l.kmeans_labels = self.kmeans_fit.labels_[start:end]
            ind += dist

        if max_clusters is None:
            max_clusters = np.amax(self.kmeans_fit.labels_)

        for l in self.layers:
            l.plot_kmeans(max_clusters)

    def plot_kmeans_wt(self, max_clusters=None):
        ind = 0
        for l in self.layers:
            dist = len(l.wavelets)
            start = ind
            end = ind + dist
            l.kmeans_labels = self.kmeans_fit.labels_[start:end]
            ind += dist

        if max_clusters is None:
            max_clusters = np.amax(self.kmeans_fit.labels_)

        for l in self.layers:
            l.plot_kmeans(max_clusters)

    def plot_fuzzy_kmeans(self, max_clusters=None):
        ind = 0
        for l in self.layers:
            dist = len(l.tsbm)
            start = ind
            end = ind + dist
            l.kmeans_labels = self.kmeans_fit.labels_[start:end]
            ind += dist

        if max_clusters is None:
            max_clusters = np.amax(self.kmeans_fit.labels_)

        for l in self.layers:
            l.plot_fuzzy_kmeans(max_clusters=max_clusters)
