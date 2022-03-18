import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from math_utils import sigmoid
from matrix_utils import shuffle


class LogisticRegression:
    def __init__(self, nb_iter=150, learning_rate=0.1, lr_ratio=1, solver='bgd', shuffle=None):
        self.nb_iter = nb_iter
        self.learning_rate = learning_rate
        self.lr_ratio = lr_ratio
        self.solver = solver
        self.shuffle = shuffle
        self.W = None
        self.b = None
        self.classes = None
        self.cost_history = None

    # 1. model F
    def __initialisation(self, nb_of_features):
        np.random.seed(0)
        W = np.random.randn(nb_of_features, 1)
        b = np.random.randn(1)
        return (W, b)

    def __H0_calculation(self, X, W, b):
        Z = np.dot(X, W) + b
        H0 = sigmoid(Z)
        return H0

    # 2. Fonction cout (== log_loss)
    def __cost_function(self, y, H0):
        m = len(y)
        epsilon = 1e-15 # to avoid errors for log(0)
        return -1/m * np.sum(y * np.log(H0 + epsilon) + (1 - y) * np.log(1 - H0 + epsilon))

    def __cost_function_stoch(self, y, H0):
        epsilon = 1e-15 # to avoid errors for log(0)
        return -(y * np.log(H0 + epsilon) + (1 - y) * np.log(1 - H0 + epsilon))

    # 3. Gradient descent
    def __grad(self, X, y, H0):
        m = len(y)
        dW = 1/m * np.dot(X.T, H0 - y)
        db = 1/m * np.sum(H0 - y)
        return dW, db

    def __grad_stoch(self, X, y, H0):
        dW = np.dot(X.T, H0 - y)
        db = H0 - y
        return dW, db

    def __BGD(self, X, y, W, b):
        cost_history = np.zeros((self.nb_iter, 1))
        for i in range(self.nb_iter):
            H0 = self.__H0_calculation(X, W, b)
            dW, db = self.__grad(X, y, H0)
            W = W - self.learning_rate * dW
            b = b - self.learning_rate * db
            cost_history[i] = self.__cost_function(y, H0)
        return W, b, cost_history

    def __SGD(self, X, y, W, b):
        m = len(y)
        cost_history = np.zeros((m * self.nb_iter, 1))
        for e in range(self.nb_iter):
            if (self.shuffle is not None):
                X, y = shuffle(X, y)
            for i in range(m):
                H0 = self.__H0_calculation(X[i:i + 1], W, b)
                dW, db = self.__grad_stoch(X[i:i + 1], y[i:i + 1], H0)
                W = W - self.learning_rate * self.lr_ratio * dW
                b = b - self.learning_rate * self.lr_ratio * db
                cost_history[i] = self.__cost_function_stoch(y[i:i + 1], H0)
        return W, b, cost_history

    def __MGD(self, X, y, W, b):
        batch_size = 10
        nb_batchs = y.shape[0] // batch_size
        cost_history = np.zeros((nb_batchs * self.nb_iter, 1))
        for e in range(1, self.nb_iter + 1):
            if (self.shuffle is not None):
                X, y = shuffle(X, y)
            for i, n in zip(range(0, y.shape[0], batch_size), range(nb_batchs)):
                H0 = self.__H0_calculation(X[i:batch_size + i], W, b)
                dW, db = self.__grad(
                    X[i:batch_size + i], y[i:batch_size + i], H0)
                W = W - self.learning_rate * dW
                b = b - self.learning_rate * db
                cost_history[n *
                             e] = self.__cost_function(y[i:batch_size + i], H0)
        return W, b, cost_history

    gradient_descent = {
        'bgd': __BGD,
        'sgd': __SGD,
        'mgd': __MGD,
    }

    # 4. Regression
    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes = np.unique(y)
        nb_of_class = self.classes.shape[0]

        y = y.reshape(y.shape[0], 1)
        self.W = np.zeros((nb_of_class, X.shape[1]))
        self.b = np.zeros((1, nb_of_class))

        if (len(self.classes) == 2):
            y = np.where(y == self.classes[1], 1, 0) # Not robust --> depends on self.classes order !!!
            self.W, self.b = self.__initialisation(X.shape[1])
            self.W, self.b, self.cost_history = self.gradient_descent[self.solver](
                self, X, y, self.W, self.b)
        else:
            for curr_class, i in zip(self.classes, range(nb_of_class)):
                W_class, b_class = self.__initialisation(X.shape[1])
                y_class = np.where(y == curr_class, 1, 0)
                W_class, b_class, cost_history_class = self.gradient_descent[self.solver](
                    self, X, y_class, W_class, b_class)
                self.W[i] = W_class.T
                self.b[0:, i] = b_class
                if self.cost_history is None:
                    self.cost_history = cost_history_class
                else:
                    self.cost_history = np.concatenate(
                        (self.cost_history, cost_history_class), axis=0)

        return self.cost_history

    # # 5. Prediction
    # def predict(self, X, weights=None, intercept=None, classes=None):

    #     if(weights is not None):
    #         self.W = weights.T
    #     if(intercept is not None):
    #         self.b = intercept

    #     H0 = None
    #     for W_class, b in zip(self.W, self.b.T):
    #         H0_class = self.__H0_calculation(X, W_class, b)
    #         H0_class = H0_class.reshape(H0_class.shape[0], 1)
    #         if H0 is None:
    #             H0 = H0_class
    #         else:
    #             H0 = np.concatenate([H0, H0_class], axis=1)
    #     if classes is not None:
    #         self.classes = classes

    #     y_pred = self.classes[np.argmax(H0, axis=1)]
    #     return y_pred
    
    def predict(self, X):
        H0 = self.__H0_calculation(X, self.W, self.b)
        y_pred = np.where(H0 >= 0.5, self.classes[1], self.classes[0])
        return y_pred.ravel()

    def accuracy_score(y_true, y_pred):
        score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
        diff = (y_pred == y_true).value_counts()
        print(diff)
        return score

    def save_to_csv(self, features, mean, std):

        nbr_of_features = len(features)
        # Create a dataframe with W
        Wb_dict = {feature: self.W[:, i] for feature, i in zip(
            features, range(nbr_of_features))}

        # Add bias, mean and std to dataframe
        for feature, i in zip(features, range(nbr_of_features)):
            Wb_dict[feature] = np.append(Wb_dict[feature], self.b[0, i])
            Wb_dict[feature] = np.append(Wb_dict[feature], mean.loc[feature])
            Wb_dict[feature] = np.append(Wb_dict[feature], std.loc[feature])

        col = np.append(self.classes, ['b', 'mean', 'std'])
        weights = pd.DataFrame.from_dict(Wb_dict, orient='index', columns=col)
        weights.index.name = 'weights'
        weights.to_csv('weights.csv')
        print(weights[self.classes])

    def plot_cost_history(self, figsize=(9, 6), mode='all', ncols=2):

        if (mode == 'class'):
            nb_of_classes = len(self.classes)
            nb_of_history = self.cost_history.shape[0]
            batch_size = int(nb_of_history / nb_of_classes)
            nrows = nb_of_classes // ncols + int(nb_of_classes % ncols > 0)
            plt.figure(figsize=figsize)
            plt.subplots_adjust(hspace=0.5, wspace=0.3)
            plt.suptitle('Learning curves')
            for i in range(1, nb_of_classes + 1):
                plt.subplot(nrows, ncols, i)
                plt.plot(
                    self.cost_history[batch_size * (i - 1):batch_size * i])
                plt.xlabel('n_iteration')
                plt.ylabel('Log_loss')
                plt.title(self.classes[i - 1])
        else:
            plt.figure(figsize=figsize)
            plt.plot(self.cost_history)
            plt.xlabel('n_iteration')
            plt.ylabel('Log_loss')
            plt.title('Evolution des erreurs')
