import numpy as np

STUDENT_NAME = 'Akshay Pala, Oghenerukevwe Ahwin, Uche McDaniel Elekwa'
STUDENT_ID = '20469485, 20868644, 20923853'

def predict(network, input_row):
    outputs = forward_propagate(network, input_row)
    # return outputs.index(max(outputs))
    return outputs

# Forward propagate input to a network output
def forward_propagate(network, input_row):
    inputs = input_row
    for n_layer in network:
        new_inputs = []
        for neuron in n_layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
            inputs = new_inputs
    return inputs

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def test_mlp(data_file):
	# Load the test set
	# START
	
    # END

	# Load your network
	# START
	trained_network = np.load("/Users/akshaypala/Documents/ECE657/weights_from_training.npy", allow_pickle=True)
	# END


	# Predict test set - one-hot encoded
	# Make a prediction with a network

	y_pred = []
	for row in X_test:
		prediction = predict(trained_network, row)
    	predicted_value = np.argmax(prediction)
    	predicted = [0 for i in range(n_outputs)]
    	predicted[predicted_value] = 1
    	y_pred.append(predicted)
   y_predicted = np.array(y_pred)
   return y_predicted


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''