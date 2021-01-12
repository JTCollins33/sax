import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, RepeatVector, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
from layers_gather import get_layers, get_validation_layers
from layer import Layer
import random


dir_path = "./data/"
fpath = "./data/featurization_figs/RAE/"
models_path = "./keras_model_progress/"
N_CLUSTERS=4
num_epochs=10
save_freq = 10
lr = 0.001
training=True
testing=False


# def build_keras_RNN_model(max_val):
#     embedding_input = tf.keras.Input((None, ))
#     input_shape = embedding_input.shape
#     embedding_sequence = tf.keras.layers.Embedding(input_dim=max_val+1, output_dim=20, mask_zero=True)(embedding_input)
#     lstm_output = tf.keras.layers.LSTM(100, activation="relu")(embedding_sequence)
#     dense_output = tf.keras.layers.Dense(10)(lstm_output)

#     model = tf.keras.Model(inputs=embedding_input, outputs = dense_output)
#     return model


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


def build_keras_RNN_model_2(max_val, test=False):
    mask_input = tf.keras.Input((None, ))
    # masked_output = tf.keras.layers.Masking(mask_value=0.)(mask_input)
    masked_output = Embedding(input_dim=max_val+1, output_dim=5, mask_zero=True, name='embedding_layer')(mask_input)
    # masked_output = Embedding(input_dim=max_val+1, output_dim=5)(mask_input)
    # masked_output = tf.squeeze(masked_output, 2)
    lstm_1_output = LSTM(5, activation='relu', name='encoder_output')(masked_output)
    repeat_vector_output = RepeatVector(K.shape(mask_input)[1], name='repeat_vector_layer')(lstm_1_output)
    lstm_2_output = LSTM(25, activation='relu', return_sequences=True, name='lstm2_output')(repeat_vector_output)
    dense_output = Dense(1, name='decoder_output')(lstm_2_output)

    if test:
        model = tf.keras.Model(inputs=mask_input, outputs=[lstm_1_output, dense_output])
    else:
        model = tf.keras.Model(inputs=mask_input, outputs=dense_output)
    return model


def prepare_dataset(layers):
    all_layers=[]
    max_lengths = [0]*len(layers)
    max_vals = [0]*len(layers)

    for i in range(len(layers)):
        for th in layers[i].curves:
            if(np.amax(th.curve)>max_vals[i]):
                max_vals[i]=np.amax(th.curve)
            if(len(th.curve)>max_lengths[i]):
                max_lengths[i]=len(th.curve)

    
    print("Formatting dataset....\n")
    for i in range(len(layers)):
        layer_list = []
        for th in layers[i].curves:            
            curve_max = np.amax(th.curve)
            zeros = np.zeros(max_lengths[i]-len(th.curve), dtype=th.curve.dtype)
            padded_arr = np.append(th.curve, zeros)
            normalized_arr = np.divide(padded_arr, 1.0*curve_max)
            # layer_list.append(np.reshape(normalized_arr, (len(normalized_arr), 1)))
            layer_list.append(normalized_arr)
        all_layers.append(np.array(layer_list))
    
    return all_layers, int(max(max_vals))


def loss_function(model, criterion, input):
    output = model(input)[:,:,0]
    loss_val = criterion(input, input)
    return loss_val

def grad(input, criterion, model):
    input = tf.convert_to_tensor(input)
    with tf.GradientTape() as tape:
        loss_value = loss_function(model, criterion, input)
    gradient = tape.gradient(loss_value, model.trainable_variables)
    gradient2 = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, gradient)]
    return loss_value, gradient2

def plot_losses(losses):
    plt.plot(losses)
    plt.savefig(fpath+"loss_plot_"+str(num_epochs)+"_epochs.png")


#find most recent model
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

    print("Preparing dataset....\n")
    all_layers, max_val = prepare_dataset(layers)


    model = build_keras_RNN_model_2(max_val, test=False)
    model.summary()
    optim = RMSprop(learning_rate=lr, clipnorm=5)
    model.compile(optimizer=optim, loss='mse')
    # criterion = tf.keras.losses.MeanSquaredError(reduction="sum")
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    losses = []

    # model.compile(optimizer=optimizer, loss=criterion)


    if(training):
        for epoch in range(1, num_epochs+1):
            random.shuffle(indices)
            epoch_total_loss = 0.0
            epoch_sum_loss = 0.0
            for i in indices:
                print("\nLayer "+str(i)+"\n")
                layer = all_layers[i]
                output_layer = layer.reshape(layer.shape[0], layer.shape[1], 1)

                history1 = model.fit(layer, output_layer, epochs=1, verbose=2)

                epoch_sum_loss += history1.history['loss'][0]

            #     loss, grads = grad(layer, criterion, model)

            #     optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #     epoch_total_loss += loss

            # epoch_avg_loss = epoch_total_loss/(1.0*len(indices))
            # losses.append(epoch_avg_loss)
            epoch_total_loss = epoch_sum_loss/(1.0*len(indices))
            losses.append(epoch_total_loss)

            print("Epoch "+str(epoch)+"/"+str(num_epochs)+"\tLoss: "+str(epoch_total_loss)+"\n")

            #save weights every save_freq epochs
            if (epoch%save_freq==0 or epoch==num_epochs):
                print("Saving model...\n")
                model.save_weights(models_path+"weights_"+str(epoch)+".hdf5")

        plot_losses(losses)


    if(testing):
        model = build_keras_RNN_model_2(max_val, test=True)
        model.load_weights(get_newest_model())

        for i in range(len(layers)):
            features = model(all_layers[i])[0]
            layers[i].set_dt_features(features)

            dt(layers[i])


    # output = model.predict(layer_array, verbose=2)
