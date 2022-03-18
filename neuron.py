from cProfile import label
from hashlib import new
from tkinter.messagebox import NO
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from math_utils import sigmoid
from matrix_utils import shuffle, standardize_df
from tqdm import tqdm


class Neuron:
    def __init__(self, nb_iter=150, learning_rate=0.1, lr_ratio=1, solver='bgd', shuffle=None):
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
    def __initialisation(self, nb_of_features):
        np.random.seed(0)
        W = np.random.randn(nb_of_features, 1)
        b = np.random.randn(1)
        return (W, b)

    def __model(self, X, W, b):
        Z = np.dot(X, W) + b
        A = sigmoid(Z)
        return A
    
    def _predict(self, X, W, b):
        A = self.__model(X, W, b)
        return A >= 0.5
    
    def predict(self, X, W, b):
        A = self.__model(X, W, b)
        y_pred = np.where(A >= 0.5, self.classes[1], self.classes[0])
        return y_pred

    # 2. Fonction cout (== log_loss)
    def __log_loss(self, y, H0):
        m = len(y)
        epsilon = 1e-15 # to avoid errors for log(0)
        return -1/m * np.sum(y * np.log(H0 + epsilon) + (1 - y) * np.log(1 - H0 + epsilon))

    # 3. Gradient descent
    def __grad(self, X, y, A):
        m = len(y)
        dW = 1/m * np.dot(X.T, A - y)
        db = 1/m * np.sum(A - y)
        return (dW, db)

    def __update(self, W, b, dW, db):
        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db
        return (W, b)

    def __BGD(self, X_train, y_train, X_test, y_test):
        W, b = self.__initialisation(X_train.shape[1])

        for i in tqdm(range(self.nb_iter)):
            A = self.__model(X_train, W, b)
            dW, db = self.__grad(X_train, y_train, A)
            W, b = self.__update(W, b, dW, db)
            # if (i % 10 == 0):
            curr_loss = self.__log_loss(y_train, A)
            self.loss.append(curr_loss)
            y_pred = self._predict(X_train, W, b)
            self.acc.append(accuracy_score(y_train, y_pred))
            if (X_test is not None and X_train is not None):
                A_test = self.__model(X_test, W, b)
                val_loss = self.__log_loss(y_test, A_test)
                self.test_loss.append(val_loss)
                y_pred = self._predict(X_test, W, b)
                acc = accuracy_score(y_test, y_pred)
                self.test_acc.append(acc)
                print(f'epoch {i}/{self.nb_iter} - loss: {curr_loss:.4f} - val_loss: {val_loss:.4f}')
            else:
                print(f'epoch {i}/{self.nb_iter} - loss: {curr_loss:.4f}')
        return W, b

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

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        y_train = y_train.reshape(y_train.shape[0], 1)
        
        self.classes = np.unique(y_train)

        if (X_test is not None and y_test is not None):
            X_test = np.asarray(X_test)
            y_test = np.asarray(y_test)
            y_test = y_test.reshape(y_test.shape[0], 1)
            y_test = np.where(y_test == self.classes[1], 1, 0) # Not robust --> depends on self.classes order !!!


        # convert str target to [0,1]
        y_train = np.where(y_train == self.classes[1], 1, 0) # Not robust --> depends on self.classes order !!!

        W, b = self.gradient_descent[self.solver](self, X_train, y_train, X_test, y_test)

        return W, b


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
                plt.subplot(1, 2, 1)
                plt.plot(self.loss, label='train loss')
                if (len(self.test_loss) != 0):
                    plt.plot(self.test_loss, label='test loss')
                    plt.legend()
                plt.xlabel('n_iteration')
                plt.ylabel('Log_loss')
                plt.title('Learning curve')
                plt.subplot(1, 2, 2)
                plt.plot(self.acc, label='train acc')
                if (len(self.test_acc) != 0):
                    plt.plot(self.test_acc, label='test acc')
                    plt.legend()
                plt.xlabel('n_iteration')
                plt.ylabel('Accuracy')
                plt.title('Accuracy')

from sklearn.datasets import make_blobs

def test_with_plant_dataset():
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape(y.shape[0], 1)

    neuron = Neuron(nb_iter=100)
    W, b = neuron.fit(X, y)
    y_pred = neuron._predict(X, W, b)
    
    print(accuracy_score(y, y_pred))
    print(W, b)

    # plt.figure(figsize=(9,6))
    # plt.scatter(X[:,0], X[:,1], c=y, cmap='winter')

    # new_plant = np.array([2, 1])
    # plt.scatter(new_plant[0], new_plant[1], c='r')

    # x0 = np.linspace(-1, 4, 100)
    # x1 = (-W[0] * x0 - b) / W[1]
    # plt.plot(x0, x1, c='orange', lw=3)
    # y_pred = neuron._predict(new_plant, W, b)
    # print(y_pred)
    neuron.plot_cost_history()

    plt.show()

from utilities import *
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


    X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
    X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()
    # print(X_train_reshape.max())

    neuron = Neuron(nb_iter=10000, learning_rate=0.01)    
    neuron.fit(X_train_reshape, y_train, X_test_reshape, y_test)
    neuron.plot_cost_history(acc=True)
  
    
    # plt.figure(figsize=(16,8))
    # for i in range(1, 10):
    #     plt.subplot(4, 5, i)
    #     plt.imshow(X_train[i], cmap='gray')
    #     plt.title(y_train[i])
    #     plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    test_with_plant_dataset()
    # test_cats_and_dogs()

