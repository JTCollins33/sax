import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
from AE import Autoencoder
from layers_gather import get_layers
from layer import Layer


dir_path = "./data/"
models_path = "./model_progress/"
fpath = "./data/featurization_figs/AE/"
N_CLUSTERS=4

#find most recent model
def get_newest_model():
    files = os.listdir(models_path)
    paths = [os.path.join(models_path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

#perform agglomerative clustering
def dt(layer):
    print("Performing clustering....\n")
    layer.agglomerative_clustering(n_clusters=N_CLUSTERS)

    print("Plotting results....\n")
    layer.plot_dt(max_clusters=N_CLUSTERS)


    layer.dt_features = np.array(layer.dt_features)
    print("Calculating cluster probabilities....\n")
    layer.fuzzy_kmeans_AE(clusters=N_CLUSTERS, standardize=True)


    print("Plotting frequency graphs....\n")
    layer.plot_kmeans_freqs(fpath+"kmeans_frequencies/")


def kmeans(layer):
    #convert into appropriate format first
    layer.dt_features = np.array(layer.dt_features)

    print("Performing kmeans....\n")
    # layer.kmeans_AE(clusters=N_CLUSTERS, standardize=True)
    layer.fuzzy_kmeans_AE(clusters=N_CLUSTERS, standardize=True)


    print("Plotting results....\n")
    layer.plot_kmeans(max_clusters=N_CLUSTERS)
    # layer.plot_kmeans_freqs()


def tsne(layer):
    print("Performing tsne....\n")
    layer.tsne_AE()

    print("Plotting results....\n")
    # layer.plot_tsne()


#method to perform testing
def test(AE_model, layers, curve_size):
    for layer in layers:
        layer_list=[]
        for th in layer.curves:
                #normalize data
                max = np.amax(th.curve)
                layer_list.append((th.curve)/(1.0*max))

        #reshape into proper tensor
        layer_tensor = torch.reshape(torch.tensor(layer_list), (len(layer_list), 1, curve_size))

        #get features from encoder
        mids = AE_model.encoder_pt1(layer_tensor).view(-1, 1, 144)

        features = AE_model.encoder_pt2(mids)

        #reshape into proper size
        features = torch.reshape(features, (features.shape[0], features.shape[2]))

        #rescale features
        # min = torch.min(features).item()
        # features = torch.add(features, (-1.0*min))
        # max = torch.max(features).item()
        # features = torch.div(features, max)

        features = features.tolist()

        #set features in layer
        print("Setting decision tree features....\n")
        layer.set_dt_features(features)

        dt(layer)
        # kmeans(layer)
        # tsne(layer)




    # for layer in layers:
    #     #find max number of peaks
    #     n_peaks=[]
    #     max_n_peaks = 0
    #     for th in layer.curves:
    #         if th.num_peaks > max_n_peaks:
    #             max_n_peaks=th.num_peaks
    #         n_peaks.append(th.num_peaks)
        
    #     n_peaks = np.array(n_peaks)
    #     n_peaks = np.true_divide(n_peaks, (1.0*(max_n_peaks)))
    #     n_peaks = (n_peaks*(4-1))


    #     plt.figure(figsize=(8,6))
    #     plt.scatter(layer.tsne_fit[:,0], layer.tsne_fit[:,1], c=n_peaks)

    # name = 'tsne.png'
    # plt.savefig(fpath+ name, bbox_inches='tight', dpi=1200)



if __name__ == '__main__':
    #get data
    layers = get_layers(dir_path, fpath, resize=True)
    curve_size = len(layers[0].curves[0].curve)

    #load in pre-trained model
    AE = torch.load(get_newest_model())

    test(AE, layers, curve_size)
