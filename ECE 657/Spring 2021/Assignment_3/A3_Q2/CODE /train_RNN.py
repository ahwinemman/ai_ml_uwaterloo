# import required packages
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout, RepeatVector,
                                     TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import load_model


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

# This function splits the sequence by using the latest 3 days
def sequenceSplit(sequence, n_steps):
    X = []
    y = []
    for i in range(len(sequence)):
        # find the end of this pattern then check if it exceeds the sequence
        p_end = i + n_steps
        if p_end > len(sequence)-1:
            break
        # split the input and targets of the sequence
        seq_x, seq_y = sequence.iloc[i:p_end,
                                     2:6], sequence.iloc[p_end, [3, 6]]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


if __name__ == "__main__":
    # 1. load your training data
    path = 'data/q2_dataset.csv'
    df_data = pd.read_csv(path)
    df_copy = df_data[::-1]
    df_copy.index = df_copy.index.values[::-1]
    df_stock = copy.deepcopy(df_copy)
    df_stock['row_num'] = np.arange(len(df_copy))

    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = sequenceSplit(df_stock, n_steps)
    # Print the data summary
    for i in range(len(X)):
        print(X[i], y[i])

    # Split the dataset here into the train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0)
    print(len(X_train), 'X train examples')
    print(len(X_test), 'X test examples')
    print(len(y_train), 'y train examples')
    print(len(y_test), 'y test examples')

    # Code to save CSV files go here

    # Arrays are converted to 2D arrays and saved in their respective csv files
    # trainReshape = X_train.reshape(X_train.shape[0], -1)
    # testReshape = X_test.reshape(X_test.shape[0], -1)

    # trainReshape_label = y_train.reshape(y_train.shape[0], -1)
    # testReshape_label = y_test.reshape(y_test.shape[0], -1)
    # # Save reshaped array to file
    # train_path = "data/train_data_RNN.csv"
    # test_path = "data/test_data_RNN.csv"
    # np.savetxt(train_path, trainReshape)
    # np.savetxt(test_path, testReshape)

    # train_labels_path = "data/train_data_RNN_labels.csv"
    # test_labels_path = "data/test_data_RNN_labels.csv"
    # np.savetxt(train_labels_path, trainReshape_label)
    # np.savetxt(test_labels_path, testReshape_label)



    # 2. Train your network
    # Make sure to print your training loss within training to show progress
    # Make sure you print the final training loss

    n_features = 4
    n_steps = 3

    

    # load train data from CSV
    train_path = "data/train_data_RNN.csv"
    train_labels_path = "data/train_data_RNN_labels.csv"
    DT = np.loadtxt(train_path)
    DTlabels = np.loadtxt(train_labels_path)

    # reshaping the array to  3D matrices.
    X_shape = np.zeros(shape=(879, 3, 4))
    X_train_ddata = DT.reshape(DT.shape[0], DT.shape[1] // X_shape.shape[2], X_shape.shape[2])

    Y_train_ddata = np.asarray(DTlabels[:, 0]).astype('float32')

    # Create model here
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(n_steps, n_features), return_sequences=True))
    
    # model.add(LSTM(64, return_sequences=False))
    # model.add(Dense(units=1, activation='linear'))
    # model.compile(optimizer='adam',loss = 'mse')
    # model.summary()
    
    # define model
    model = Sequential()
    model.add(GRU(40, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss = 'mse')
    model.summary()
    

    # fit model
    history = model.fit(X_train_ddata, Y_train_ddata,   epochs=8000, batch_size=32)

    # Plot the loss on a graph here
    plt.title('Cross Entropy Loss')
    plt.xlabel('Time (epochs)')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], color='red', label='Training  loss')
    plt.legend(loc='upper right')
    plt.show()


    # 3. Save your model
    # model_path =  "data/20923853_RNN_model.h5"
    # model.save(model_path)


