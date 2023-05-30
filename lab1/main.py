from data import generate_linear, generate_XOR_easy, show_result, plot_loss
from layer import FCLayer, ActivationLayer, Convolutional, Reshape
import numpy as np
import warnings

def train_test_spilt(x, y, train_ratio):
    return x[0:int(x.shape[0]*train_ratio)][:], y[0:int(x.shape[0]*train_ratio)][:], x[int(x.shape[0]*train_ratio):][:], y[int(x.shape[0]*train_ratio):][:]
    # return x, y, x, y

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def ReLU(x):
    return x * (x > 0)

def ReLU_prime(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size



class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i].T
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(list(output[0]))

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        loss_record = []
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j].T
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate[int(i/(int(epochs/len(learning_rate))+1))])

            # calculate average error on all samples
            err = err / samples
            loss_record.append(err)
            if (i+1) % 1000 == 0:
                print(f'epoch {i+1}/{epochs}   loss={err}')
        return loss_record


# training data
# x, y = generate_linear()
x, y = generate_XOR_easy()
x_train, y_train, x_test, y_test = train_test_spilt(x, y, 0.8)




# network
net = Network()
net.add(FCLayer(2, 4, 'sgd'))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(4, 4, 'sgd'))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
# net.add(ActivationLayer(sigmoid, sigmoid_prime))  
#add convolutional layer with reshape layer
# net.add(Reshape((9,1), (3,3)))
# net.add(Convolutional((3,3), 2)) # output 2 * 2
# net.add(Reshape((2,2), (4,1)))
# net.add(FCLayer(4, 4, 'sgd'))
# net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(4, 1, 'sgd'))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(mse, mse_prime)
loss_record = net.fit(np.expand_dims(x_train, axis=1), np.expand_dims(y_train, axis=1), epochs=10000, learning_rate=[0.5, 0.2, 0.1])
plot_loss(loss_record)

# test
out = net.predict(np.expand_dims(x_test, axis=1))
prediction_loss = mse(y_test, out)
pred_y = []
acc = 0
for idx, i in enumerate(out):
    print(f'Iter{idx:4} |    Ground truth: {y_test[idx][0]} |    prediction: {i[0]:.5f} |')
    if round(i[0]) == y_test[idx][0]:
        acc += 1
    pred_y.append(round(i[0]))
show_result(x_test, y_test, pred_y)

print(f'loss= {prediction_loss:.5f} accuracy={acc/len(y_test)*100}%')





