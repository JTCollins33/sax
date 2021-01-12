import h5py
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import sklearn.cluster as cluster
from sklearn.mixture import GaussianMixture
# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import sklearn.decomposition as decomp
import sklearn.manifold as man
import sklearn.preprocessing as pp

WAVELET_LENGTH=20
N_DT_FEATURES=3

class Layer:
    def __init__(self, curves, points, index=None, figs=None):
        if curves is not None:
            self.curves = sorted(curves, key=operator.attrgetter('index'))
        else:
            self.curves = None
        self.points = points
        self.tsne_fit = None
        self.pca_fit = None
        self.kmeans_fit = None
        self.kmeans_labels = None
        self.tsbm = None
        self.wavelets = None
        self.distance = None
        self.vector = None
        self.index = index
        self.figs = figs
        self.dt_features = None
        self.dt_labels = None
        self.dt_fit = None

    def _gather_tsbm(self):
        d = []
        for c in self.curves:
            d.append(c.tsbm)
        return np.array(d)

    def _gather_wavelets(self):
        d=[]
        min=len(self.curves[0].wavelets)
        for c in self.curves:
            if (len(c.wavelets)<min):
                min=len(c.wavelets)
        for c in self.curves:
            # wave = c.shrink_around_max(min)
            wave = c.center_around_max(21)
            d.append(wave)
        return np.array(d)

    def compute_wt(self):
        max_size = 0
        total = len(self.curves)
        counter = 0
        incr = total / 20
        prog = 1
        for c in self.curves:
            #generate alphabetical strings
            c.compute_wt()
            if counter > prog:
                progress = int((counter / total) * 100.0)
                print('Computing Wavelet Transformations | {}% Complete'.format(progress))
                prog += incr
            counter += 1
            if (len(c.wavelets)>max_size):
                max_size=len(c.wavelets)
                
        #make all curves the same length
        # print("Fixing wavelet lengths...\n")
        # for c in self.curves:
        #     c.fix_wt_length(max_size)
        # print("Done fixing wavelet lengths\n")
        self.wavelets = self._gather_wavelets()

    def compute_sax(self):
        total = len(self.curves)
        counter = 0
        incr = total / 20
        prog = 1
        for c in self.curves:
            #generate alphabetical strings
            c.compute_sax()
            if counter > prog:
                progress = int((counter / total) * 100.0)
                print('Computing SAX Representations | {}% Complete'.format(progress))
                prog += incr
            counter += 1


    def compute_tsbm(self):
        total = len(self.curves)
        counter = 0
        incr = total / 20
        prog = 1
        for c in self.curves:
            #convert count arrays from alphabetical strings (skipping some transitions; ex: ae is only ae count)
            c.compute_tsbm()
            if counter > prog:
                progress = int((counter / total) * 100.0)
                print('Computing TSBM Representations | {}% Complete'.format(progress))
                prog += incr
            counter += 1
        self.tsbm = self._gather_tsbm()

    def compute_tsbm_ivt(self):
        total = len(self.curves)
        counter = 0
        incr = total / 20
        prog = 1
        for c in self.curves:
            #convert count arrays from alphabetical strings (including intermediate steps; ex: ac would increment ab and bc counts)
            c.compute_tsbm_ivt()
            if counter > prog:
                progress = int((counter / total) * 100.0)
                print('Computing TSBM Representations | {}% Complete'.format(progress))
                prog += incr
            counter += 1
        self.tsbm = self._gather_tsbm()

    def save_sax(self, path):
        f = h5py.File(path, 'a')
        grp = f.create_group('SAX')
        total = len(self.curves)
        counter = 0
        incr = total / 20
        prog = 1
        for c in self.curves:
            dt = 'S{}'.format(len(c.sax))
            dset = grp.create_dataset(str(c.index), shape=(1,), dtype=dt)
            dset[...] = np.string_(c.sax)
            if counter > prog:
                progress = int((counter / total) * 100.0)
                print('Writing SAX Representations | File: {} | {}% Complete'.format(os.path.basename(path), progress))
                prog += incr
            counter += 1
        f.close()

    def save_tsbm(self, path):
        f = h5py.File(path, 'a')
        grp = f.create_group('TSBM')
        grp.create_dataset("TSBM", data=self.tsbm)
        f.close()

    def read_sax(self, path):
        f = h5py.File(path, 'r')
        total = len(f['SAX'].keys())
        counter = 0
        incr = total / 20
        prog = 1
        for n, d in f['SAX'].items():
            self.curves[int(n)].sax = d.value.tostring().decode('UTF-8')
            if counter > prog:
                progress = int((counter / total) * 100.0)
                print('Reading SAX Representations | File: {} | {}% Complete'.format(os.path.basename(path), progress))
                prog += incr
            counter += 1
        f.close()

    def read_tsbm(self, path):
        f = h5py.File(path, 'r')
        tsbm = f['TSBM/TSBM'].value
        self.tsbm = tsbm
        f.close()

    #this gets the specified features for the decision tree clustering 
    #FEATURES: ths max and # temps above 250
    def get_dt_features(self):
        self.dt_features = np.zeros((len(self.curves), N_DT_FEATURES))

        print("Calculating decision tree features...\n")
        #get key features for each thermal curve
        for i in range(len(self.curves)):
            # self.dt_features[i][0], self.dt_features[i][1], self.dt_features[i][2], self.dt_features[i][3] = self.curves[i].get_dt_features(threshold=1000.0)
            self.dt_features[i][0], self.dt_features[i][1], self.dt_features[i][2] = self.curves[i].get_dt_features(threshold=1000.0)
        print("Done calculating decision tree features.\n")

    #this method sets the decision tree features found in auto-encoder
    def set_dt_features(self, features):
        self.dt_features = features
        

    def agglomerative_clustering(self):
        model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
        self.dt_fit = model.fit(self.dt_features)
        self.dt_labels = model.labels_

    def pca(self, comps=2):
        pca_func = decomp.PCA(n_components=comps)
        self.pca_fit = pca_func.fit_transform(self.tsbm)

    def tsne(self, comps=2, perplexity=175, early_exaggeration=50.0, learning_rate=250.0):
        tsne_func = man.TSNE(n_components=comps, perplexity=perplexity, early_exaggeration=early_exaggeration,
                             learning_rate=learning_rate)
        self.tsne_fit = tsne_func.fit_transform(self.tsbm)

    def kmeans(self, clusters=5, standardize=True):
        mask = np.all(self.tsbm == 0, axis=0)
        d = self.tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.KMeans().fit(d)
        self.kmeans_labels = self.kmeans_fit.labels_

    def kmeans_AE(self, clusters=5, standardize=True):
        mask = np.all(self.dt_features==0, axis=0)
        d = self.dt_features[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.KMeans().fit(d)
        self.kmeans_labels = self.kmeans_fit.labels_


    def kmeans_wt(self, clusters=5, standardize=True):
        mask = np.all(self.wavelets == 0, axis=0)
        d = self.wavelets[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.KMeans().fit(d)
        self.kmeans_labels = self.kmeans_fit.labels_

    def dbscan(self, eps=0.5, min_samples=5, standardize=True):
        mask = np.all(self.tsbm == 0, axis=0)
        d = self.tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(d)

    def kmedoids(self, clusters=5, standardize=True):
        mask = np.all(self.tsbm == 0, axis=0)
        d = self.tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = KMedoids(n_clusters=clusters).fit(d)
    
    def mean_shift(self, clusters=5, standardize=True):
        mask = np.all(self.tsbm == 0, axis=0)
        d = self.tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = cluster.MeanShift(n_clusters=clusters).fit(d)
        self.kmeans_labels = self.kmeans_fit.labels_

    def fuzzy_kmeans(self, clusters=5, standardize=True):
        mask = np.all(self.tsbm == 0, axis=0)
        d = self.tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = ske.fuzzy_kmeans.FuzzyKMeans(c=clusters).fit(d)

    def EM(self, clusters=5, standardize=True):
        mask = np.all(self.tsbm == 0, axis=0)
        d = self.tsbm[:, np.invert(mask)]
        if standardize:
            d = pp.StandardScaler().fit_transform(d)
        self.kmeans_fit = GaussianMixture(n_components=clusters).predict(d)

    def plot_tsne(self):
        plt.clf()
        plt.scatter(self.tsne_fit[:, 0], self.tsne_fit[:, 1])
        plt.show()

    def plot_dt(self, max_clusters=None):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        cmap = plt.cm.get_cmap('rainbow', max_clusters)

        plt.scatter(self.points[:, 0], self.points[:, 1], c=self.dt_labels, cmap=cmap, s=3)

        cbar = plt.colorbar(ticks=range(max_clusters))
        cbar.set_label('Cluster', fontsize='large', fontweight='bold', rotation=270, labelpad=15)
        plt.xlabel('X (mm)', fontsize='large', fontweight='bold')
        plt.ylabel('Y (mm)', fontsize='large', fontweight='bold')
        if max_clusters is not None:
            plt.clim(0, max_clusters - 1)
        tick_locs = (np.arange(max_clusters) + 0.5)*(max_clusters - 1) / max_clusters
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(np.arange(max_clusters))
        plt.axes().set_aspect('equal', 'datalim')
        name = 'dt_' + str(self.index) + '.png'
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)


    def plot_pca(self, max_clusters=None):
        plt.clf()
        fig = plt.figure()
        ax=fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        cmap = plt.cm.get_cmap('rainbow', max_clusters)

        plt.scatter(self.pca_fit[:, 0], self.pca_fit[:, 1], c=self.group_dist[:,0], cmap='rainbow', s=1)

        cbar = plt.colorbar()
        cbar.set_label('Distance from Vector Start (mm)', fontsize='large', fontweight='bold', rotation=270,
                       labelpad=15)
        plt.ylabel('Principal Component 2', fontsize='large', fontweight='bold')
        plt.xlabel('Principal Component 1', fontsize='large', fontweight='bold')
        name = 'pca' + '.png'
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)

    def plot_kmeans(self, max_clusters=None, suffix=''):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        cmap = plt.cm.get_cmap('rainbow', max_clusters)
        # plt.scatter(self.points[:, [0]], self.points[:, [1]], c=self.kmeans_labelscmap=cmap, s=3)

        """Remove this part later """
        if (len(self.kmeans_labels) > len(self.points)):
            arr = []
            for i in range(len(self.points)):
                arr.append(self.kmeans_labels[i])
            self.kmeans_labels=np.array(arr)
        elif (len(self.kmeans_labels) < len(self.points)):
            arr = []
            for i in range(len(self.points)):
                if i>=len(self.kmeans_labels):
                    arr.append(0)
                else:
                    arr.append(self.kmeans_labels[i])
            self.kmeans_labels=np.array(arr)


        """This is where it ends """

        plt.scatter(self.points[:, 0], self.points[:, 1], c=self.kmeans_labels, cmap=cmap, s=3)

        cbar = plt.colorbar(ticks=range(max_clusters))
        cbar.set_label('Cluster', fontsize='large', fontweight='bold', rotation=270, labelpad=15)
        plt.xlabel('X (mm)', fontsize='large', fontweight='bold')
        plt.ylabel('Y (mm)', fontsize='large', fontweight='bold')
        if max_clusters is not None:
            plt.clim(0, max_clusters - 1)
        tick_locs = (np.arange(max_clusters) + 0.5)*(max_clusters - 1) / max_clusters
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(np.arange(max_clusters))
        plt.axes().set_aspect('equal', 'datalim')
        name = 'kmeans' + str(self.index) + suffix + '.png'
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)

    #this method plots max_clusters number of graphs, one for each cluster's probability
    def plot_fuzzy_kmeans(self, max_clusters=None, suffix=''):
        #create directory for images
        folder = "ThermalHistories_"+str(self.index)+"/"
        if(not os.path.isdir(self.figs+folder[0:len(folder)-1])):
            os.mkdir(self.figs+folder[0:len(folder)-1])
        
        fig_names = ["Cluster_0", "Cluster_1", "Cluster_2", "Cluster_3"]
        for i in range(max_clusters):
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)
            ax.tick_params(width=2, labelsize='large')
            cmap = plt.cm.get_cmap('rainbow', max_clusters)
            plt.scatter(self.points[:, [0]], self.points[:, [1]], c=self.kmeans_labels[:, [i]], cmap='gray', s=3)


            """Remove this part later """
            # if (len(self.kmeans_labels) > len(self.points)):
            #     arr = []
            #     for i in range(len(self.points)):
            #         arr.append(self.kmeans_labels[i])
            #     self.kmeans_labels=np.array(arr)
            # elif (len(self.kmeans_labels) < len(self.points)):
            #     arr = []
            #     for i in range(len(self.points)):
            #         if i>=len(self.kmeans_labels):
            #             arr.append(0)
            #         else:
            #             arr.append(self.kmeans_labels[i])
            #     self.kmeans_labels=np.array(arr)

            """This is where it ends """

            # plt.scatter(self.points[:, 0], self.points[:, 1], c=self.kmeans_labels, cmap=cmap, s=3)

            # cbar = plt.colorbar(ticks=range(max_clusters))
            # cbar.set_label('Cluster', fontsize='large', fontweight='bold', rotation=270, labelpad=15)
            plt.title(fig_names[i])
            plt.xlabel('X (mm)', fontsize='large', fontweight='bold')
            plt.ylabel('Y (mm)', fontsize='large', fontweight='bold')
            if max_clusters is not None:
                plt.clim(0, max_clusters - 1)
            tick_locs = (np.arange(max_clusters) + 0.5)*(max_clusters - 1) / max_clusters
            # cbar.set_ticks(tick_locs)
            # cbar.set_ticklabels(np.arange(max_clusters))
            plt.axes().set_aspect('equal', 'datalim')
            name = 'fuzzy_kmeans' + str(self.index)+"_" + suffix + fig_names[i]+'.png'
            plt.savefig(self.figs+folder+name, bbox_inches='tight', dpi=1200)

    def plot_tsbm(self, index):
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
        mask = np.all(self.tsbm == 0, axis=0)
        d = self.tsbm[index, np.invert(mask)]
        plt.xticks(range(np.count_nonzero(np.invert(mask))), labels[np.invert(mask)])
        plt.ylabel('Frequency', fontsize='large', fontweight='bold')
        name = 'single_tsbm' + '.png'
        color = iter(plt.cm.rainbow(np.linspace(0, 1, cc.shape[0])))
        for i in range(cc.shape[0]):
            c = next(color)
            label = 'Cluster ' + str(i)
            plt.plot(cc[[i], :][0], label=label, linewidth=3, marker='o', linestyle='--', c=c)
        ax.legend()
        # plt.plot(d, linewidth=3, marker='o', linestyle='--')
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)

    def plot_distance(self):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(width=2, labelsize='large')
        plt.ylabel('Temperature ($^\circ$C)', fontsize='large', fontweight='bold')
        plt.xlabel('Time (s)', fontsize='large', fontweight='bold')
        plt.axes().set_aspect('equal', 'datalim')
        plt.scatter(self.points[:, [0]], self.points[:, [1]], c=self.distance)
        name = 'layer' + str(self.index) + '.png'
        plt.savefig(self.figs + name, bbox_inches='tight', dpi=1200)

    def plot_vector(self):
        plt.clf()
        plt.scatter(self.points[:, [0]], self.points[:, [1]], c=self.vector)
        plt.show()
