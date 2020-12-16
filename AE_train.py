import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from AE import Autoencoder
from layers_gather import get_layers
from layer import Layer
from AE_test import test


dir_path = "./data/"
fpath = "./data/featurization_figs/AE/8_features/"
loss_plots_path = "./loss_plots/"
n_epochs = 50
learning_rate = 0.001
TEST = False

if __name__ == '__main__':
    #setup data
    layers = get_layers(dir_path, fpath, resize=True)
    curve_size = len(layers[0].curves[0].curve)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    #setup model
    AE = Autoencoder().to(device)

    #setup loss function
    lossF = nn.MSELoss()

    #setup optimizers
    optim = torch.optim.Adam(AE.parameters(), lr=learning_rate, weight_decay=1e-5)


    indices = []
    losses = []
    epochs = []
    for i in range(len(layers)):
        indices.append(i)

    print("Starting autoencoder training...\n")
    for epoch in range(1, n_epochs+1):
        random.shuffle(indices)
        for i in indices:
            layer = []
            for th in layers[i].curves:
                #normalize data
                max = np.amax(th.curve)
                layer.append((th.curve)/(1.0*max))

            #reshape into proper tensor
            layer = torch.reshape(torch.tensor(layer), (len(layer), 1, curve_size))
            
            #run encoder and decoder
            generated_layer = AE(layer)

            #calculate loss and backpropogate
            loss = lossF(generated_layer, layer)
            loss.backward()

            #update weights in network
            optim.step()
            optim.zero_grad()

            #save losses for plotting
            losses.append(float(loss))
            epochs.append(epoch)

        print("Epoch: "+str(epoch)+", Loss: "+str(float(loss)))

        # save training progress every 10 epochs
        if(epoch%10==0 or epoch==n_epochs):
            if isinstance(AE, nn.Module):
                torch.save(AE, "./model_progress/AE_model_"+str(epoch)+"_epochs.pth")


    #plot training losses
    plt.plot(epochs, losses)
    plt.xlabel("Epoch Number")
    plt.ylabel("Autoencoder Loss")
    plt.title("Autoencoder Loss Over "+str(n_epochs)+" Epochs")
    plt.savefig(loss_plots_path+"AE_loss_"+str(n_epochs)+"_epochs.png")

    #if desire to test in training script, do so
    if(TEST):
        test(AE, layers, curve_size)