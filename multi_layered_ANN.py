from bisect import bisect
from utilities import *
from sklearn.datasets import make_circles
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from math_utils import sigmoid
from matrix_utils import shuffle, standardize_df
from tqdm import tqdm
# from scipy.special import softmax
from sklearn.utils.extmath import softmax
from sklearn.neural_network import MLPClassifier

# def softmax(x):
#     e = np.exp(x - np.max(x))
#     return e / np.sum(e, axis=1, keepdims=True)

def relu(x):
    return np.maximum(x, 0)

class Neuron:
    def __init__(self, layers=[32, 21, 12], activation='relu',  nb_iter=150, learning_rate=0.1, lr_ratio=1, solver='bgd', shuffle=None):
        self.layers = layers
        self.nb_iter = nb_iter
        self.learning_rate = learning_rate
        self.lr_ratio = lr_ratio
        self.solver = solver
        self.shuffle = shuffle
        self.W = None
        self.b = None
        self.classes = None
        self.loss = []
        self.acc = []
        self.test_loss = []
        self.test_acc = []

    # 1. model F
    def __initialisation(self, layers):
        
        n_layers = len(layers)
        parameters = {}
        np.random.seed(0)

        for i in range(1, n_layers):
            parameters[f'W{i}'] = np.random.randn(layers[i], layers[i-1])
            parameters[f'b{i}'] = np.random.randn(layers[i], 1)

        # parameters = {
        #     f'C{i}': {
        #         'W': np.random.randn(layers[i], layers[i-1]),
        #         'b': np.random.randn(layers[i], 1)
        #     } for i in range(1, n_layers)
        # }
        return parameters

    def __forward_propagation(self, X, parameters: dict):

        # A = X.copy() # should we use a copy of X or not ?
        activations = {'A0': X}
        for i in range(1, len(parameters) // 2 + 1):
            Z = np.dot(parameters[f'W{i}'], activations[f'A{i-1}']) + parameters[f'b{i}']
            if (i != len(parameters) // 2):
                activations[f'A{i}'] = sigmoid(Z)
        
        # For the output layer
        # t = np.exp(Z)
        activations[f'A{len(parameters) // 2}'] = softmax(Z)

        return activations

    def _predict(self, X, parameters):
        activations = self.__forward_propagation(X, parameters)
        out = len(parameters) // 2
        A = activations[f'A{out}']
        return A >= 0.5
    
    def predict(self, X, parameters):
        activations = self.__forward_propagation(X, parameters)
        out = len(parameters) // 2
        A = activations[f'A{out}']
        y_pred = np.where(A >= 0.5, self.classes[1], self.classes[0])
        return y_pred.ravel()
    
    def predict_softmax(self, X, parameters):
        activations = self.__forward_propagation(X, parameters)
        out = len(parameters) // 2
        y_pred = activations[f'A{out}']
        # y_pred = np.where(A >= 0.5, self.classes[1], self.classes[0])
        return y_pred

    # 2. Fonction cout (== log_loss)
    def __log_loss(self, y, H0):
        m = len(y)
        epsilon = 1e-15  # to avoid errors for log(0)
        return -1/m * np.sum(y * np.log(H0 + epsilon) + (1 - y) * np.log(1 - H0 + epsilon))

    # 2. Fonction cout (== log_loss for soft max)
    # def __log_loss(self, y, H0):
    #     m = len(y)
    #     epsilon = 1e-15  # to avoid errors for log(0)
    #     return -1/m * np.sum(y * np.log(H0 + epsilon))

    # 3. Gradient descent
    def __back_propagation(self, X, y, activations, parameters):

        m = y.shape[1]
        out = len(parameters) // 2

        dZ = activations[f'A{out}'] - y
        # dW_out = 1/m * np.dot(dZ, activations[f'A{out - 1}'].T)
        # db_out = 1/m * np.sum(dZ, axis=1, keepdims=True)

        gradients = {}
        for i in reversed(range(1, out + 1)):
            gradients[f'dW{i}'] = 1/m * np.dot(dZ, activations[f'A{i-1}'].T)
            gradients[f'db{i}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if i > 1:
                dZ = np.dot(parameters[f'W{i}'].T, dZ) * activations[f'A{i-1}'] * (1 - activations[f'A{i-1}'])

        # dZ = np.dot(parameters['W2'].T, dZ) * activations['A1'] * (1 - activations['A1'])
        # gradients[f'dW1'] = 1/m * np.dot(dZ, X.T)
        # gradients[f'db1'] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        return gradients

    def __update(self, gradients, parameters):

        for i in range(1, len(parameters) // 2 + 1):
            parameters[f'W{i}'] = parameters[f'W{i}'] - self.learning_rate * gradients[f'dW{i}']
            parameters[f'W{i}'] = parameters[f'W{i}'] - self.learning_rate * gradients[f'db{i}']

        return parameters

    def __BGD(self, X_train, y_train, X_test, y_test):

        n0 = X_train.shape[0]  # /!\ we use the T of X so n = X.shape[0]
        # n2 = y_train.shape[0]  # /!\ we use the T of y so n = y.shape[0]
        n2 = len(self.classes)  # nbr of classes = nbr of output neurons

        layers = [n0] + self.layers + [n2]

        parameters = self.__initialisation(layers)
        # exit()

        for i in tqdm(range(self.nb_iter)):
            activations = self.__forward_propagation(X_train, parameters)
            gradients = self.__back_propagation(
                X_train, y_train, activations, parameters)
            parameters = self.__update(gradients, parameters)
            # print('len', len(activations[f'A{len(activations) - 1}']))
            # print(activations[f'A{len(activations) - 1}'])
            self.loss.append(self.__log_loss(y_train, activations[f'A{len(activations) - 1}']))
            y_pred = self._predict(X_train, parameters)
            # self.acc.append(accuracy_score(
            #     y_train.flatten(), y_pred.flatten()))
            # if (X_test is not None and X_train is not None):
            #     A_test = self.__forward_propagation(X_test, parameters)
            #     self.test_loss.append(
            #         self.__log_loss(y_test, A_test[f'A{len(A_test)}']))
            #     y_pred = self._predict(X_test, parameters)
            #     self.test_acc.append(accuracy_score(
            #         y_test.flatten(), y_pred.flatten()))
        return parameters

    def __SGD(self, X, y, W, b):
        m = len(y)
        cost_history = np.zeros((m * self.nb_iter, 1))
        for e in range(self.nb_iter):
            if (self.shuffle is not None):
                X, y = shuffle(X, y)
            for i in range(m):
                H0 = self.__model(X[i:i + 1], W, b)
                dW, db = self.__grad_stoch(X[i:i + 1], y[i:i + 1], H0)
                W = W - self.learning_rate * self.lr_ratio * dW
                b = b - self.learning_rate * self.lr_ratio * db
                cost_history[i] = self.__log_loss_stoch(y[i:i + 1], H0)
        return W, b, cost_history

    def __MGD(self, X_train, y_train, X_test, y_test):
        batch_size = 10
        nb_batchs = y_train.shape[1] // batch_size

        print('nb_batchs', nb_batchs)

        n0 = X_train.shape[0]  # /!\ we use the T of X so n = X.shape[0]
        n2 = y_train.shape[0]  # /!\ we use the T of y so n = y.shape[0]

        layers = [n0] + self.layers + [n2]

        parameters = self.__initialisation(layers)

        for e in range(1, self.nb_iter + 1):
            if (self.shuffle is not None):
                X_train, y_train = shuffle(X_train, y_train)
            for i in tqdm(range(0, y_train.shape[1], batch_size)):
                activations = self.__forward_propagation(X_train[:, i:batch_size + i], parameters)
                gradients = self.__back_propagation(
                    X_train[:, i:batch_size + i], y_train[:, i:batch_size + i], activations, parameters)
                parameters = self.__update(gradients, parameters)
            
                # log_loss de sklearn
                self.loss.append(self.__log_loss(y_train[:, i:batch_size + i], activations[f'A{len(activations) - 1}']))
                y_pred = self._predict(X_train, parameters)
                self.acc.append(accuracy_score(
                    y_train.flatten(), y_pred.flatten()))
                if (X_test is not None and X_train is not None):
                    A_test = self.__forward_propagation(X_test, parameters)
                    self.test_loss.append(
                        self.__log_loss(y_test, A_test[f'A{len(A_test)}']))
                    y_pred = self._predict(X_test, parameters)
                    self.test_acc.append(accuracy_score(
                        y_test.flatten(), y_pred.flatten()))
        return parameters


    gradient_descent = {
        'bgd': __BGD,
        'sgd': __SGD,
        'mgd': __MGD,
    }

    # 4. Regression
    def fit(self, X_train, y_train, X_test=None, y_test=None):

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        y_train = y_train.reshape(1, y_train.shape[0])
        
        self.classes = np.unique(y_train)

        if (X_test is not None and y_test is not None):
            X_test = np.asarray(X_test)
            y_test = np.asarray(y_test)
            y_test = y_test.reshape(1, y_test.shape[0])
            y_test = np.where(y_test == self.classes[1], 1, 0) # Not robust --> depends on self.classes order !!!


        # convert str target to [0,1]
        y_train = np.where(y_train == self.classes[1], 1, 0) # Not robust --> depends on self.classes order !!!


        parameters = self.gradient_descent[self.solver](
            self, X_train, y_train, X_test, y_test)
        return parameters

    def plot_cost_history(self, figsize=(9, 6), acc=False):

            if acc == False:
                plt.figure(figsize=figsize)
                plt.plot(self.loss)
                plt.xlabel('n_iteration')
                plt.ylabel('Log_loss')
                plt.title('Learning curve')
            else:
                plt.figure(figsize=figsize)
                plt.suptitle('Learning curves and accuracy')

                plt.subplot(1, 3, 1)
                plt.plot(self.loss, label='train loss')
                plt.xlabel('n_iteration')
                plt.ylabel('Log_loss')
                plt.legend()
                
                if (len(self.test_loss) != 0):
                    plt.subplot(1, 3, 2)
                    plt.plot(self.test_loss, label='val loss')
                    plt.xlabel('n_iteration')
                    plt.ylabel('Log_loss')
                    plt.legend()
                
                plt.subplot(1, 3, 3)
                plt.plot(self.acc, label='train acc')
                plt.xlabel('n_iteration')
                plt.ylabel('Accuracy')
                plt.title('Accuracy')
                if (len(self.test_acc) != 0):
                    plt.plot(self.test_acc, label='val acc')
                plt.legend()


def test_with_circle_dataset():

    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))

    print('dimensions de X:', X.shape)
    print('dimensions de y:', y.shape)

    plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
    plt.show()

    neuron = Neuron(nb_iter=1000, n1=16, learning_rate=0.1)
    neuron.fit(X, y)
    neuron.plot_cost_history(acc=True)

    plt.show()


def test_cats_and_dogs():
    X_train, y_train, X_test, y_test = load_data()

    # # Plot the pictures with their class
    # plt.figure(figsize=(16,8))
    # for i in range(1, 10):
    #     plt.subplot(4, 5, i)
    #     plt.imshow(X_train[i], cmap='gray')
    #     plt.title(y_train[i])
    #     plt.tight_layout()

    # Transpose the data and reshape in 2d to use it in neuron
    y_train = y_train.T
    y_test = y_test.T

    X_train = X_train.T
    X_train_reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()

    X_test = X_test.T
    X_test_reshape = X_test.reshape(-1, X_test.shape[-1]) / X_test.max()

    # # these nbr to limit the nbr of pictures to work with
    # m_train = 300
    # m_test = 80

    # X_train_reshape = X_train_reshape[:, :m_train]
    # X_test_reshape = X_test_reshape[:, :m_test]
    # y_train = y_train[:, :m_train]
    # y_test = y_test[:, :m_test]

    print("TRAIN SET")
    print(X_train_reshape.shape)
    print(y_train.shape)
    print(np.unique(y_train, return_counts=True))
    print("TEST SET")
    print(X_test_reshape.shape)
    print(y_test.shape)
    print(np.unique(y_test, return_counts=True))

    # set neural network and launch the training
    layers = [32, 16]
    neuron = Neuron(layers, nb_iter=2, learning_rate=0.01, solver='mgd')
    neuron.fit(X_train_reshape, y_train, X_test_reshape, y_test)
    neuron.plot_cost_history(acc=True)

    plt.show()


if __name__ == '__main__':

    try:
        # test_with_circle_dataset()
        test_cats_and_dogs()
    except KeyboardInterrupt:
        print ('Simulation aborted by ctrl-c')



