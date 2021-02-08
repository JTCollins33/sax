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
from torch.autograd import Variable



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
    def __init__(self, num_features, hidden_size, num_layers, num_classes, device):
        super(REncoder, self).__init__()
        self.hidden_size=hidden_size
        self.device = device
        self.r1 = nn.GRU(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.r2 = nn.GRU(input_size=hidden_size, hidden_size=N_FEATURES, batch_first=True)
        # self.emb = nn.Embedding(num_embeddings=20, embedding_dim=input_size)
        # self.r1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, input, lens, hidden):
        x = nn.utils.rnn.pack_padded_sequence(input, lens.cpu(), batch_first=True).to(self.device)
        x, hidden = self.r1(x, hidden)
#         x, hidden = self.r1(x, None)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[:,-1,:]
        out = self.lin1(x)
        return out, hidden

class RDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, num_classes, device):
        super(RDecoder, self).__init__()
        self.hidden_size=hidden_size
        self.device = device
        self.r1 = nn.GRU(input_size=num_classes, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.r2 = nn.GRU(input_size=hidden_size, hidden_size=output_size, num_layers=num_layers, batch_first=True)
        self.lin1 = nn.Linear(hidden_size, output_size)


    def forward(self, input, dim, hidden):
        #simulate repeat_vector
        x = torch.zeros([input.shape[0], dim, input.shape[1]], dtype = input[0][0].dtype).to(self.device)
        for i in range(x.shape[0]):
            for j in range(dim):
                x[i][j] = input[i]

        x, hidden = self.r1(x, hidden)
#         x, hidden = self.r1(x, None)
        x = self.lin1(x)
        return x, hidden



class RAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_classes, device):
        super(RAE, self).__init__()
        self.hidden_size=hidden_size
        self.device = device
        self.encoder = REncoder(input_size, hidden_size, num_layers, num_classes, device)
        self.decoder = RDecoder(output_size, hidden_size, num_layers, num_classes, device)

    def forward(self, input, dim, lens, hidden):
        features, encoder_hidden = self.encoder(input, lens, hidden)
        decoder_hidden = encoder_hidden[:1]
        output, decoder_hidden = self.decoder(features, dim, decoder_hidden)
        return output.to(self.device), decoder_hidden[:1].to(self.device)

    def initHidden(self):
        return Variable(torch.zeros([1, batch_size, self.hidden_size], dtype=torch.float), requires_grad=True).to(self.device)




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

def sort_batch(layers):
    sorted = []
    while (len(layers)>0):
        max_length=0
        max_index=-1
        for i in range(len(layers)):
            if(len(layers[i].curve)>max_length):
                max_length = len(layers[i].curve)
                max_index = i
        sorted.append(layers.pop(max_index))
    return sorted



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #get layer
    layers = get_layers(dir_path, fpath, resize=False)

    indices = []
    for i in range(len(layers)):
        indices.append(i)

    """
    Define Model
    """
    model = RAE(1, 5, 1, 1, N_FEATURES, device).to(device)
    criterion = nn.MSELoss()
    optim = optim.Adam(model.parameters(), lr=lr)


    """
    Start Training
    """
    if(training):
        x=1
        losses = []
        hidden=model.initHidden()
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

                    # ths_batch = layer.curves[batch_base:batch_base+batch_size]
                    ths_batch= sort_batch(layer.curves[batch_base:batch_base+batch_size])
                    batch_list = []
                    seq_lens = []
                    max_length = 0
                    for th in ths_batch:
                        seq_lens.append(len(th.curve))
                        if (len(th.curve)>max_length):
                            max_length = len(th.curve)

                    #make curves in batch fixed length and normalize them
                    for th in ths_batch:
                        curve_max = np.amax(th.curve)
                        zeros = np.zeros(max_length-len(th.curve), dtype=th.curve.dtype)
                        padded_arr = np.append(th.curve, zeros)
                        normalized_arr = np.divide(padded_arr, 1.0*curve_max)
                        # int_arr = normalized_arr.astype(np.int64)
                        batch_list.append(normalized_arr)
                    batch = torch.tensor(batch_list)

                    batch = batch.reshape(batch.shape[0], batch.shape[1], 1).to(device)

                    #pass into model
                    output, hidden = model(batch, batch.shape[1], torch.tensor(seq_lens).to(device), hidden)

                    #compute loss and backpropagate
                    loss = criterion(batch, output)
                    loss.backward(retain_graph=True)
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
