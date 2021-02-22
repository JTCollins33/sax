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
from layers_gather import get_layers, get_validation_layers
from layer import Layer
from AE_test import test


dir_path = "./data/"
fpath = "./data/featurization_figs/"
loss_plots_path = "./loss_plots/"
n_epochs = 1000
learning_rate = 0.001
ngpu = 0
TEST = True

if __name__ == '__main__':
    #setup data
    layers = get_layers(dir_path, fpath, resize=True)
    curve_size = len(layers[0].curves[0].curve)

    validation_layers = get_validation_layers(dir_path, fpath, resize=True)

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
    validation_losses = []
    epochs = []
    for i in range(len(layers)):
        indices.append(i)

    print("Starting autoencoder training...\n")
    epoch = 1
    while epoch < n_epochs+1:
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

        #check overfitting
        if(epoch%5==0):
            val_layer = []
            for th in validation_layers[0].curves:
                max = np.amax(th.curve)
                val_layer.append((th.curve)/(1.0*max))

            val_layer = torch.reshape(torch.tensor(val_layer), (len(val_layer), 1, curve_size))
            generated_val_layer = AE(val_layer)
            validation_losses.append(float(lossF(generated_val_layer, val_layer)))

            n_val_losses = len(validation_losses)
            if (n_val_losses>1):
                #if model starts to overfit, save model and exit training
                if ((validation_losses[n_val_losses-1] > validation_losses[n_val_losses-2]) and (validation_losses[n_val_losses-2]>validation_losses[n_val_losses-3])):
                    torch.save(AE, "./model_progress/AE_model_"+str(epoch)+"_epochs.pth")
                    epoch = n_epochs

        # save training progress every 10 epochs
        if(epoch%10==0 or epoch==n_epochs):
            if isinstance(AE, nn.Module):
                torch.save(AE, "./model_progress/AE_model_"+str(epoch)+"_epochs.pth")

        epoch +=1


    #plot training losses
    plt.plot(epochs, losses)
    plt.xlabel("Epoch Number")
    plt.ylabel("Autoencoder Loss")
    plt.title("Autoencoder Loss Over "+str(n_epochs)+" Epochs")
    plt.savefig(loss_plots_path+"AE_loss_"+str(n_epochs)+"_epochs.png")

    #if desire to test in training script, do so
    if(TEST):
        test(AE, layers, curve_size)
