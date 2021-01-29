import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from layers_gather import get_layers, get_validation_layers
from layer import Layer
import random


dir_path = "./data/"
fpath = "./data/featurization_figs/RAE/new/"
models_path = "./pytorch_model_progress/"
N_CLUSTERS=4
num_epochs=100
save_freq=2
batch_size=15
lr = 0.001
N_FEATURES=8
training=True
testing=False


class REncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(REncoder, self).__init__()
        # self.r1 = nn.GRU(input_size=None, hidden_size=None, batch_first=True)
        # self.r2 = nn.GRU(input_size=None, hidden_size=None, batch_first=True)
        self.emb = nn.Embedding(num_embeddings=20, embedding_dim=input_size)
        self.r1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        x = self.emb(input)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[3]))
        x, (h_n, h_c) = self.r1(x, None)
        x = self.lin1(x[:, -1, :])
        return x

class RDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, num_classes):
        super(RDecoder, self).__init__()
        self.r1 = nn.LSTM(input_size=num_classes, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.r2 = nn.LSTM(input_size=hidden_size, hidden_size=output_size, num_layers=num_layers, batch_first=True)


    def forward(self, input, dim):
        # with torch.no_grad():
        #     x = nn.Linear(N_FEATURES, dim)(input)
        #     x = nn.Linear(dim, dim*N_FEATURES)(x)

        #simulate repeat_vector
        x = torch.zeros([input.shape[0], dim, input.shape[1]], dtype = input[0][0].dtype)
        for i in range(x.shape[0]):
            for j in range(dim):
                x[i][j] = input[i]

        # x = torch.reshape(x, (x.shape[0], dim, N_FEATURES))
        x, (h_n, h_c) = self.r1(x, None)
        x, (h_n, h_c) = self.r2(x, None)
        return x

class RAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_classes):
        super(RAE, self).__init__()
        self.encoder = REncoder(input_size, hidden_size, num_layers, num_classes)
        self.decoder = RDecoder(output_size, hidden_size, num_layers, num_classes)

    def forward(self, input, dim):
        features = self.encoder(input)
        output = self.decoder(features, dim)
        return output




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


def get_max(layers):
    max_vals = [0]*len(layers)
    for i in range(len(layers)):
        for th in layers[i].curves:
            if(np.amax(th.curve)>max_vals[i]):
                max_vals[i]=np.amax(th.curve)

    return int(max(max_vals))



def prepare_dataset(layers):
    all_layers=[]
    max_lengths = [0]*len(layers)
    
    max_val = get_max(layers)

    print("Formatting dataset....\n")
    for i in range(len(layers)):
        layer_list = []
        for th in layers[i].curves:            
            curve_max = np.amax(th.curve)
            zeros = np.zeros(max_lengths[i]-len(th.curve), dtype=th.curve.dtype)
            padded_arr = np.append(th.curve, zeros)
            normalized_arr = np.divide(padded_arr, 1.0*curve_max)
            layer_list.append(normalized_arr)
        all_layers.append(np.array(layer_list))
    
    return all_layers, max_val

def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(fpath+"loss_plot_"+str(num_epochs)+"_epochs.png")

#find most recent saved model
def get_newest_model():
    files = os.listdir(models_path)
    paths = [os.path.join(models_path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

if __name__ == "__main__":
    #get layer
    layers = get_layers(dir_path, fpath, resize=False)

    indices = []
    for i in range(len(layers)):
        indices.append(i)

    """
    Define Model
    """
    model = RAE(5, 5, 1, 1, N_FEATURES)
    criterion = nn.MSELoss()
    optim = optim.Adam(model.parameters(), lr=lr)


    """
    Start Training
    """
    if(training):
        x=1
        losses = []
        for epoch in range(1, num_epochs+1):
            random.shuffle(indices)
            epoch_total_loss = 0.0
            epoch_sum_loss = 0.0
            for i in indices:
                layer = layers[i]
                batch_base=0
                layer_loss_sum=0.0
                batch_cnt = 0
                start_time = time.time()
                done = False

                #create batch for testing
                while(not(done)):
                    while(batch_base+batch_size>=len(layer.curves)):
                        batch_base-=1
                        done=True

                    ths_batch = layer.curves[batch_base:batch_base+batch_size]
                    batch_list = []
                    max_length = 0
                    for th in ths_batch:
                        if (len(th.curve)>max_length):
                            max_length = len(th.curve)

                    #make curves in batch fixed length and normalize them
                    for th in ths_batch:
                        curve_max = np.amax(th.curve)
                        zeros = np.zeros(max_length-len(th.curve), dtype=th.curve.dtype)
                        padded_arr = np.append(th.curve, zeros)
                        normalized_arr = np.divide(padded_arr, 1.0*curve_max)
                        int_arr = normalized_arr.astype(np.int64)
                        batch_list.append(int_arr)
                    # batch = np.array(batch_list)
                    batch = torch.tensor(batch_list)

                    # output_batch = batch.reshape(batch.shape[0], batch.shape[1], 1)
                    batch = batch.reshape(batch.shape[0], batch.shape[1], 1)

                    #pass into model
                    output = model(batch, batch.shape[1])

                    #compute loss and backpropagate
                    loss = criterion(batch, output)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

                    layer_loss_sum += float(loss)
                    batch_cnt += 1
                
                    #move to next batch
                    batch_base+=batch_size

                layer_loss = layer_loss_sum/(1.0*batch_cnt)
                print("Layer "+str(i)+"\tLoss: "+str(layer_loss)+"\tTime: "+str(time.time()-start_time)+" seconds\n")

                epoch_sum_loss += layer_loss

            epoch_total_loss = epoch_sum_loss/(1.0*len(indices))
            losses.append(epoch_total_loss)
            print("Epoch "+str(epoch)+"/"+str(num_epochs)+"\tLoss: "+str(epoch_total_loss)+"\n")

            if(epoch%save_freq==0 or epoch==num_epochs):
                torch.save(model, models_path+"model_"+str(epoch)+"_epochs")

        plot_losses(losses)



    """
    Start Testing
    """
    if(testing):
        x=2
