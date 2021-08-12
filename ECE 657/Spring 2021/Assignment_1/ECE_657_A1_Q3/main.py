import pandas as pd
from sklearn.model_selection import train_test_split
from math import exp
from random import seed
from random import random
import numpy as np
from IPython.display import Image, display

# read the training data and labels
df = pd.read_csv("/Users/akshaypala/Downloads/train_data.csv")
labels = pd.read_csv("/Users/akshaypala/Downloads/train_labels.csv")

X2 = np.array(df)
y2 = np.array(labels)
y2 = y2.astype(np.int64)
# X2 = X2.astype(np.int64)
y2 = list(y2)

# split the dataset into test and train splits
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=42)


# Initialize a network
def initialize_network(no_inputs, no_hidden, no_outputs):
    net = []
    hidden_layer = [{'weights': [random() for i in range(no_inputs + 1)]} for i in range(no_hidden)]
    net.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(no_hidden + 1)]} for i in range(no_outputs)]
    net.append(output_layer)
    return net


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    # return 1.0 / (1.0 + exp(-activation))
    return (exp(activation) - exp(-activation)) / (exp(activation) + exp(-activation))
    # return max(0.0, activation)


# Forward propagate input to a network output
def forward_propagate(net, input_row):
    inputs = input_row
    for n_layer in net:
        new_inputs = []
        for neuron in n_layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of a neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate any errors and store in neurons
def backward_propagate(net, expected):
    for i in reversed(range(len(net))):
        n_layer = net[i]
        errors = []
        if i != len(net) - 1:
            for j in range(len(n_layer)):
                error = 0.0
                for neuron in net[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(n_layer)):
                neuron = n_layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(n_layer)):
            neuron = n_layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(net, in_row, learn_rate):
    for i in range(len(net)):
        inputs = in_row
        if i != 0:
            inputs = [neuron['output'] for neuron in net[i - 1]]
        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learn_rate * neuron['delta'] * inputs[j]
            # neuron['weights'][-1] += learn_rate * neuron['delta']


# train the network with the given dataset
def train_network(net, train_data, learn_rate, no_epoch, no_outputs):
    for epoch in range(no_epoch):
        total_error = 0
        for j, row_in in enumerate(train_data):
            outputs = forward_propagate(net, row_in)
            expected = [0 for i in range(no_outputs)]
            counter = -1
            for element in y_train[j]:
                counter = counter + 1
                if element == 1:
                    expected[counter] = 1
            total_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate(net, expected)
            update_weights(net, row_in, learn_rate)
        print('>epoch=%d, learn_rate=%.3f, error=%.3f' % (epoch, learn_rate, total_error))


# Make a prediction with a network
def predict(net, input_row):
    outputs = forward_propagate(net, input_row)
    return outputs


# get the accuracy of the trained model
def accuracy(y_true, y_preds):
    if not (len(y_true) == len(y_preds)):
        print('Size of predicted and true labels not equal.')
        return 0.0

    corr = 0
    for i in range(0,len(y_true)):
        corr += 1 if (y_true[i] == y_preds[i]).all() else 0

    return corr/len(y_true)


seed(1)
# set the number of inputs and outputs
n_inputs = len(X_train[0])
n_outputs = len(y_train[0])

# initialize network with the above inputs and outputs and 1 hidden layer
network = initialize_network(n_inputs, 1, n_outputs)

# train the network with split train set, learning parameter = 0.5, number of epochs = 5
train_network(network, X_train, 0.75, 20, n_outputs)

trained_layers = []
for layer in network:
    print(layer)
    trained_layers.append(layer)
trained_layers = np.array(trained_layers)

# save the trained weights in a separate file
np.save("/Users/akshaypala/Documents/ECE657/weights_from_training", trained_layers)
# load the file for prediction
trained_network = np.load("/Users/akshaypala/Documents/ECE657/weights_from_training.npy", allow_pickle=True)

# predict class using testing set split
y_pred = []
for row in X_test:
    # prediction = predict(network, row)
    prediction = predict(trained_network, row)
    predicted_value = np.argmax(prediction)
    predicted = [0 for i in range(n_outputs)]
    predicted[predicted_value] = 1
    y_pred.append(predicted)
y_predicted = np.array(y_pred)
print(y_predicted)

# calculate and print score
score = accuracy(y_test, y_predicted)
print(score)
