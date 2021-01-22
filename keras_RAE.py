import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, RepeatVector, Dense, Masking
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from layers_gather import get_layers, get_validation_layers
from layer import Layer
import random


dir_path = "./data/"
fpath = "./data/featurization_figs/RAE/new/"
models_path = "./keras_model_progress/"
N_CLUSTERS=4
num_epochs=10
save_freq=2
batch_size=15
lr = 0.001
training=True
testing=True



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


def build_keras_RNN_model(max_val, test=False):
    mask_input = tf.keras.Input((None, ))
    masked_output = Embedding(input_dim=max_val+1, output_dim=5, mask_zero=True, name='embedding_layer')(mask_input)
    lstm_1_output = LSTM(5, activation='relu', name='encoder_output')(masked_output)
    repeat_vector_output = RepeatVector(K.shape(mask_input)[1], name='repeat_vector_layer')(lstm_1_output)
    lstm_2_output = LSTM(25, activation='relu', return_sequences=True, name='lstm2_output')(repeat_vector_output)
    dense_output = Dense(1, name='decoder_output', activation='softmax')(lstm_2_output)

    #make model output intermediate layer if testing (to get features)
    if test:
        model = tf.keras.Model(inputs=mask_input, outputs=[lstm_1_output, dense_output])
    else:
        model = tf.keras.Model(inputs=mask_input, outputs=dense_output)
    return model

def build_test_model(test=False):
    input = tf.keras.Input((None, 1))
    mask = Masking(mask_value=0.)(input)
    features = LSTM(7, activation='relu', name='encoder_output')(mask)
    repeat_vector_output = RepeatVector(K.shape(input)[1], name='repeat_vector_layer')(features)
    lstm_2_output = LSTM(28, activation='relu', return_sequences=True, name='lstm2_output')(repeat_vector_output)
    dense_output = Dense(1, name='decoder_output', activation='softmax')(lstm_2_output)

    if(test):
        model = tf.keras.Model(inputs=input, outputs=[features, dense_output])
    else:
        model = tf.keras.Model(inputs=input, outputs=dense_output)
    return model
    



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

    print("Preparing dataset....\n")
    # all_layers, max_val = prepare_dataset(layers)
    # max_val = get_max(layers)


    # model = build_keras_RNN_model(max_val, test=False)
    model = build_test_model(test=False)
    model.summary()
    optim = tf.keras.optimizers.RMSprop(lr=lr, clipnorm=3)
    model.compile(optimizer=optim, loss='mse')

    # #if there are previously saved weights, load them in
    # if(os.listdir(models_path) != []):
    #     model.load_weights(get_newest_model())
    
    # criterion = tf.keras.losses.MeanSquaredError(reduction="sum")
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    # model.compile(optimizer=optimizer, loss=criterion)

    losses = []

    if(training):
        for epoch in range(1, num_epochs+1):
            random.shuffle(indices)
            epoch_total_loss = 0.0
            epoch_sum_loss = 0.0
            for i in indices:
                layer = layers[i]
                batch_base=0
                layer_loss_sum=0.0

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
                        batch_list.append(normalized_arr)
                    batch = np.array(batch_list)

                    # output_batch = batch.reshape(batch.shape[0], batch.shape[1], 1)
                    batch = batch.reshape(batch.shape[0], batch.shape[1], 1)


                    # history1 = model.fit(batch, output_batch, epochs=1, verbose=0)
                    history1 = model.fit(batch, batch, epochs=1, verbose=0)


                    layer_loss_sum += history1.history['loss'][0]
                    batch_base+=batch_size
                    print("Layer "+str(i)+": Batch "+str(batch_base)+"-"+str(batch_base+batch_size)+"/"+str(len(layer.curves))+"\tLoss: "+str(history1.history['loss'][0])+"\n")

                layer_loss = layer_loss_sum/(1.0*len(layer.curves))

                print("Layer "+str(i)+"\tLoss: "+str(layer_loss)+"\tTime: "+str(time.time()-start_time)+" seconds\n")
                
                epoch_sum_loss += layer_loss
                # output_layer = layer.reshape(layer.shape[0], layer.shape[1], 1)
                # history1 = model.fit(layer, output_layer, epochs=1, verbose=2)

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
        # model = build_keras_RNN_model(max_val, test=True)
        model = build_test_model(test=True)
        model.load_weights(get_newest_model())

        for i in range(len(layers)):
            layer = layers[i]
            layer_list = []

            layer_features = []
            done = False
            batch_base = 0
            while(not(done) and batch_base<len(layer.curves)):
                start = batch_base
                end=batch_base+batch_size

                while(end>len(layer.curves)):
                    end-=1
                    done=True

                ths_batch = layer.curves[start:end]

                batch_list = []
                max_length = 0
                for th in ths_batch:
                    if (len(th.curve)>max_length):
                        max_length = len(th.curve)

                for th in ths_batch:
                    curve_max = np.amax(th.curve)
                    zeros = np.zeros(max_length-len(th.curve), dtype=th.curve.dtype)
                    padded_arr = np.append(th.curve, zeros)
                    normalized_arr = np.divide(padded_arr, 1.0*curve_max)
                    batch_list.append(normalized_arr)
                
                batch = np.array(batch_list)
                batch = np.reshape(batch, (batch.shape[0], batch.shape[1], 1))

                batch_features = model.predict(batch)[0]

                for j in range(batch_features.shape[0]):
                    layer_features.append(batch_features[j])

                batch_base += batch_size

            # #find max length curve
            # max_length=0
            # for th in layer.curves:
            #     if(len(th.curve)>max_length):
            #         max_length = len(th.curve)

            # #normalize each curve so same length
            # for th in layer.curves:
            #     curve_max = np.amax(th.curve)
            #     zeros = np.zeros(max_length-len(th.curve), dtype=th.curve.dtype)
            #     padded_arr = np.append(th.curve, zeros)
            #     normalized_arr = np.divide(padded_arr, 1.0*curve_max)
            #     layer_list.append(normalized_arr)

            # layer_arr = np.array(layer_list)
            # # layer_tensor = K.constant(layr_arr)
            # features = model.predict(layer_arr)[0]
            layers[i].set_dt_features(layer_features)

            dt(layers[i])


    # output = model.predict(layer_array, verbose=2)
