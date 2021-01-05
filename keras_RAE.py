import tensorflow as tf
from keras import backend as K
import numpy as np
from layers_gather import get_layers, get_validation_layers
from layer import Layer


dir_path = "./data/"
fpath = "./data/featurization_figs/AE/9_features/th_size_33/"


# def build_keras_RNN_model(max_val):
#     embedding_input = tf.keras.Input((None, ))
#     input_shape = embedding_input.shape
#     embedding_sequence = tf.keras.layers.Embedding(input_dim=max_val+1, output_dim=20, mask_zero=True)(embedding_input)
#     lstm_output = tf.keras.layers.LSTM(100, activation="relu")(embedding_sequence)
#     dense_output = tf.keras.layers.Dense(10)(lstm_output)

#     model = tf.keras.Model(inputs=embedding_input, outputs = dense_output)
#     return model

def build_keras_RNN_model_2(max_length):
    mask_input = tf.keras.Input((None, 1))
    masked_output = tf.keras.layers.Masking(mask_value=0.)(mask_input)
    lstm_1_output = tf.keras.layers.LSTM(10, activation='relu')(masked_output)
    repeat_vector_output = tf.keras.layers.RepeatVector(K.shape(mask_input)[1])(lstm_1_output)
    lstm_2_output = tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)(repeat_vector_output)
    dense_output = tf.keras.layers.Dense(1)(lstm_2_output)

    model = tf.keras.Model(inputs=mask_input, outputs=dense_output)
    return model


def stack_tensors(list):
    values = tf.concat(list, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in list])
    return tf.RaggedTensor.from_row_lengths(values, lens)


if __name__ == "__main__":
    #get layer
    layer = get_layers(dir_path, fpath, resize=False)

    layer_list = []
    max_val = 0
    max_length = 0

    for th in layer[0].curves:
        if np.amax(th.curve)>max_val:
            max_val = int(np.amax(th.curve))
        if len(th.curve) > max_length:
            max_length = len(th.curve)

    for th in layer[0].curves:
        zeros = np.zeros(max_length-len(th.curve), dtype = th.curve.dtype)
        padded_arr = np.append(th.curve, zeros)
        layer_list.append(np.reshape(padded_arr, (len(padded_arr), 1)))
        # curve_tensor = tf.convert_to_tensor(th.curve, dtype = th.curve.dtype)
        # layer_list.append(tf.reshape(curve_tensor, (curve_tensor.shape[0], 1)))
        # layer_list.append(curve_tensor)

    # ragged_tensor = stack_tensors(layer_list)


    layer_array = np.array(layer_list)
    # layer_array = tf.convert_to_tensor(layer_list)
    model = build_keras_RNN_model_2(max_length)
    model.summary()

    model.compile(optimizer='adam', loss='mse')
    model.fit(layer_array, layer_array, epochs=2, verbose=2)
