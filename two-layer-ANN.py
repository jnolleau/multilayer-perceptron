from utilities import *
from sklearn.datasets import make_circles
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from math_utils import sigmoid
from matrix_utils import shuffle, standardize_df
from tqdm import tqdm


class Neuron:
    def __init__(self, n1=1, nb_iter=150, learning_rate=0.1, lr_ratio=1, solver='bgd', shuffle=None):
        self.n1 = n1
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
    def __initialisation(self, n0, n1, n2):

        W1 = np.random.randn(n1, n0)
        b1 = np.random.randn(n1, 1)
        W2 = np.random.randn(n2, n1)
        b2 = np.random.randn(n2, 1)

        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        return parameters

    def __forward_propagation(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        Z1 = np.dot(W1, X) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        activations = {
            'A1': A1,
            'A2': A2
        }
        return activations

    def _predict(self, X, parameters):
        activations = self.__forward_propagation(X, parameters)
        A2 = activations['A2']
        return A2 >= 0.5

    # 2. Fonction cout (== log_loss)
    def __log_loss(self, y, H0):
        m = len(y)
        epsilon = 1e-15  # to avoid errors for log(0)
        return -1/m * np.sum(y * np.log(H0 + epsilon) + (1 - y) * np.log(1 - H0 + epsilon))

    # 3. Gradient descent
    def __back_propagation(self, X, y, activations, parameters):

        m = y.shape[1]
        A1 = activations['A1']
        A2 = activations['A2']
        W2 = parameters['W2']

        dZ2 = A2 - y
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {
            'dW1': dW1,
            'dW2': dW2,
            'db1': db1,
            'db2': db2
        }
        return gradients

    def __update(self, gradients, parameters):

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
      
        dW1 = gradients['dW1']
        db1 = gradients['db1']
        dW2 = gradients['dW2']
        db2 = gradients['db2']

        W1 = W1 - self.learning_rate * dW1
        b1 = b1- self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2- self.learning_rate * db2

        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        return parameters

    def __BGD(self, X_train, y_train, X_test, y_test):
        
        n0 = X_train.shape[0] # /!\ we use the T of X so n = X.shape[0]
        n2 = y_train.shape[0] # /!\ we use the T of y so n = y.shape[0]

        parameters = self.__initialisation(n0, self.n1, n2)

        for i in tqdm(range(self.nb_iter)):
            activations = self.__forward_propagation(X_train, parameters)
            gradients = self.__back_propagation(X_train, y_train, activations, parameters)
            parameters = self.__update(gradients, parameters)

            if (i % 100 == 0):
                self.loss.append(self.__log_loss(y_train, activations['A2'])) #log_loss de sklearn
                y_pred = self._predict(X_train, parameters)
                self.acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
                if (X_test is not None and X_train is not None):
                    A_test = self.__forward_propagation(X_test, parameters)
                    self.test_loss.append(self.__log_loss(y_test, A_test['A2']))
                    y_pred = self._predict(X_test, parameters)
                    self.test_acc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))
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

    def __MGD(self, X, y, W, b):
        batch_size = 10
        nb_batchs = len(y) // batch_size
        cost_history = np.zeros((nb_batchs * self.nb_iter, 1))
        for e in range(1, self.nb_iter + 1):
            if (self.shuffle is not None):
                X, y = shuffle(X, y)
            for i, n in zip(range(0, nb_batchs, batch_size), range(nb_batchs)):
                H0 = self.__model(X[i:batch_size + i], W, b)
                dW, db = self.__grad(
                    X[i:batch_size + i], y[i:batch_size + i], H0)
                W = W - self.learning_rate * dW
                b = b - self.learning_rate * db
                cost_history[n *
                             e] = self.__log_loss(y[i:batch_size + i], H0)
        return W, b, cost_history

    gradient_descent = {
        'bgd': __BGD,
        'sgd': __SGD,
        'mgd': __MGD,
    }

    # 4. Regression
    def fit(self, X_train, y_train, X_test=None, y_test=None):

        # X = np.asarray(X_train)
        # y = np.asarray(y_train)
        # y = y.reshape(y.shape[0], 1)

        parameters = self.gradient_descent[self.solver](self, X_train, y_train, X_test, y_test)
        return parameters

    def plot_cost_history(self, figsize=(9, 6), mode='all', ncols=2, acc=False):

        if (mode == 'class'):
            nb_of_classes = len(self.classes)
            nb_of_history = self.loss.shape[0]
            batch_size = int(nb_of_history / nb_of_classes)
            nrows = nb_of_classes // ncols + int(nb_of_classes % ncols > 0)
            plt.figure(figsize=figsize)
            plt.subplots_adjust(hspace=0.5, wspace=0.3)
            plt.suptitle('Learning curves')
            for i in range(1, nb_of_classes + 1):
                plt.subplot(nrows, ncols, i)
                plt.plot(
                    self.loss[batch_size * (i - 1):batch_size * i])
                plt.xlabel('n_iteration')
                plt.ylabel('Log_loss')
                plt.title(self.classes[i - 1])
        else:
            if acc == False:
                plt.figure(figsize=figsize)
                plt.plot(self.loss)
                plt.xlabel('n_iteration')
                plt.ylabel('Log_loss')
                plt.title('Learning curve')
            else:
                plt.figure(figsize=figsize)
                plt.subplot(1, 3, 1)
                plt.plot(self.loss, label='train loss')
                if (len(self.test_loss) != 0):
                    plt.subplot(1, 3, 2)
                    plt.plot(self.test_loss, label='test loss')
                    plt.legend()
                plt.xlabel('n_iteration')
                plt.ylabel('Log_loss')
                plt.title('Learning curve')
                plt.subplot(1, 3, 3)
                plt.plot(self.acc, label='train acc')
                if (len(self.test_acc) != 0):
                    plt.plot(self.test_acc, label='test acc')
                    plt.legend()
                plt.xlabel('n_iteration')
                plt.ylabel('Accuracy')
                plt.title('Accuracy')


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
    print("TRAIN SET")
    print(X_train.shape)
    print(y_train.shape)
    print(np.unique(y_train, return_counts=True))
    print("TEST SET")
    print(X_test.shape)
    print(y_test.shape)
    print(np.unique(y_test, return_counts=True))

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

    # these nbr to limit the nbr of pictures to work with
    m_train = 300
    m_test = 80

    X_train_reshape = X_train_reshape[:, :m_train]
    X_test_reshape = X_test_reshape[:, :m_test]
    y_train = y_train[:, :m_train]
    y_test = y_test[:, :m_test]

    # set neural network and launch the training
    neuron = Neuron(nb_iter=8000, n1=32, learning_rate=0.01)
    neuron.fit(X_train_reshape, y_train, X_test_reshape, y_test)
    neuron.plot_cost_history(acc=True)

    plt.show()


if __name__ == '__main__':
    # test_with_circle_dataset()
    test_cats_and_dogs()
    # n = [4, 2, 6 , 2, 3]
    # tup = tuple(i for i in n)
    # print(tup)
