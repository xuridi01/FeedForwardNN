import  numpy as np

def sigmoid_function(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid_function(x)*(1-sigmoid_function(x))

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = [np.random.randn(x, 1) for x in self.layers[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(self.layers[:-1], self.layers[1:])]

    def feed_forward(self, previous_activation):
        act = previous_activation

        #computing whole layer, dot - matrix of weights with vector of activation of previous layer
        for w, b in zip(self.weights, self.biases):
            act = sigmoid_function(np.dot(w, act) + b)

        #returning vector of output layer activations
        return act

    def feed_forward_activations_and_z_arrays(self, previous_activation):
        #arrays for returning activations and Zs
        activations_arr = [previous_activation]
        z_arr = []

        #computing whole layer, dot - matrix of weights with vector of activation of previous layer
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations_arr[-1]) + b
            z_arr.append(z)
            layer_activation = sigmoid_function(z)
            activations_arr.append(layer_activation)

        return activations_arr, z_arr

    @staticmethod
    def compute_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    #computing gradients for every w and b, using 4 equations of backpropagation
    def backpropagation(self, input_values, true_values):
        #preparing arrays for gradients
        w_gradients = [np.zeros(w.shape) for w in self.weights]
        b_gradients = [np.zeros(b.shape) for b in self.biases]

        #feed forward first to get activations and z from every layer
        activations_arr, z_arr = self.feed_forward_activations_and_z_arrays(input_values)

        #output layer error and gradients
        error = (activations_arr[-1] - true_values) * sigmoid_derivative(z_arr[-1])
        w_gradients[-1] = np.dot(error, activations_arr[-2].transpose())
        b_gradients[-1] = error

        #hidden layers
        for i in range(2, self.num_layers):
            error = (np.dot(self.weights[-i + 1].transpose(), error) * sigmoid_derivative(z_arr[-i]))
            w_gradients[-i] = np.dot(error, activations_arr[-i - 1].transpose())
            b_gradients[-i] = error

        return w_gradients, b_gradients

    def update_parameters(self, weight_gradients, bias_gradients, learning_rate):
        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, weight_gradients)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, bias_gradients)]

    def evaluate(self, t_data):
        correct_predictions = 0
        loss_sum = 0
        for x, y in t_data:
            output_activation = self.feed_forward(x)
            #we need to handle categorical format of output and expected output ->argmax
            correct_predictions += np.argmax(output_activation) == np.argmax(y)
            loss_sum += self.compute_loss(y, output_activation)

        return correct_predictions, loss_sum

    def get_batches(self, training_data, batch_size):
        for i in range(0, len(training_data), batch_size):
            yield training_data[i:i + batch_size]

    def train_network(self, training_data, epochs, learning_rate, t_data, batch_size):
        n_test = len(t_data)
        n_training = len(training_data)

        total_w_g = [np.zeros(w.shape) for w in self.weights]
        total_b_g = [np.zeros(b.shape) for b in self.biases]

        #training network through epochs
        for i in range(epochs):
            for j in range(0, n_training, batch_size):
                batch = training_data[j:j + batch_size]

                for input_data, expected_output in batch:
                    w_g, b_g = self.backpropagation(input_data, expected_output)

                    total_w_g += w_g
                    total_b_g += b_g

            for j in range(len(total_w_g)):
                total_w_g[j] /= n_training
                total_b_g[j] /= n_training

            self.update_parameters(total_w_g, total_b_g, learning_rate)

            #evaluating correct predictions and loss on test data
            corr, loss_sum = self.evaluate(t_data)
            loss = loss_sum / n_test
            print(f"Epoch {i+1}/{epochs} completed. Correctly classified {corr}/{n_test}. Accuracy: {100*corr/n_test} % ")
            print(f"Epoch {i+1}, Loss: {loss}")