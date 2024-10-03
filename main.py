import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import NN

# Loading the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Converting the data into NumPy arrays
images = mnist.data.values
labels = mnist.target.values.astype(int)

#categorical -> got 10 neurons and neuron with the highest value is representing the prediction
def sparse_to_categorical(label,num_classes):
    res=np.zeros((num_classes,1))
    res[label]=1
    return res

# def plot_mnist(data):
#     """Plots 16 images from the data"""
#     fig,axes=plt.subplots(4,4,figsize=(16,16))
#     for i, ax in enumerate(axes.flatten()):
#         ax.imshow(data[i][0].reshape((28,28)),cmap='gray')
#         ax.set_title(f"Label {np.argmax(data[i][1])}")
#     plt.show()

training_data = [(x.reshape(784,1) / 255.0, sparse_to_categorical(y, 10)) for x, y in zip(images, labels)]

test_data = training_data[-10000:]
training_data = training_data[:-10000]

#here we can experiment with params for learning
epochs = 5
learning_rate = 0.5
batch_size = 8

#plot_mnist(training_data[:16])

NN=NN.NeuralNetwork([784,20,10])
NN.train_network(training_data, epochs, learning_rate, test_data, batch_size)

