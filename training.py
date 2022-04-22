import argparse
import re
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from csv_reader import read_csv
from matrix_utils import standardize_df, standardize_df_test
from LR_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from multi_layered_ANN import Neuron


def log_reg_skl(X_train, X_test, y_train, y_test):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # pd.DataFrame(y_pred).to_csv(
    #     'houses.csv', header=[target], index_label='Index')

    # count nbr of true / false
    diff = (y_pred == y_test).value_counts()
    print(diff)

    # Accuracy score
    score = accuracy_score(y_true=y_test, y_pred=y_pred) * 100
    print(f"accuracy_score: {score:.3}%")
    f1 = f1_score(y_true=y_test, y_pred=y_pred, pos_label='M')
    print(f"f1_score: {f1:.2}")



def log_reg(X_train, X_test, y_train, y_test, solver):

    if solver == 'bgd':
        # Use Batch Gradient Descent
        lr = LR()
    elif solver == 'mgd':
        # Use Mini-Batch Gradient Descent
        lr = LR(nb_iter=3, solver='mgd')
    elif solver == 'sgd':
        # Use Stochastic Gradient Descent
        lr = LR(nb_iter=1, lr_ratio=0.2, solver='sgd')
    else:
        print(f'Bad solver: {solver}')
        sys.exit(1)

    print('Logistic Regression in progress...')
    lr.fit(X_train, y_train)
    print('Logistic Regression done !')

    # lr.save_to_csv(features, mean, std)
    lr.plot_cost_history()

    if X_test is not None:
        print("\nPredictions:\n============")
        y_pred = lr.predict(X_test)
        diff = (y_pred == y_test).value_counts()
        print(diff)
        # Sklearn Accuracy score
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"accuracy_score: {score:.2f}")
        f1 = f1_score(y_true=y_test, y_pred=y_pred, pos_label='M')
        print(f"f1_score: {f1:.2f}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset for training as .csv')
    parser.add_argument('--solver', '-s', help='Algorithm to use in the optimization problem',
                        default='bgd', choices=['bgd', 'sgd', 'mgd'])
    parser.add_argument('--test', '-t', help='Dataset for testing as .csv')
    parser.add_argument(
        '--compare', '-c', help='Allows comparison with sklearn LR', action="store_true")

    args = parser.parse_args()

    match = re.search(".*.csv", args.dataset)
    if match:
        df = read_csv(match.group(0))
    else:
        print(f"Bad file extension: {args.dataset}")
        sys.exit(1)

    # Remove missing values
    idx_with_missing_values = df['ID'][df['concavity1'] == 0].index
    df = df.drop(labels=idx_with_missing_values, axis=0)

    # Split dataset
    X = df.drop(['ID', 'Diagnosis'], axis=1)
    X = standardize_df(X)

    y = df['Diagnosis']
    # y = np.where(y == 'M', 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('X_train.shape', X_train.shape)

    target = 'Diagnosis'

    # Logistic Regression
    # log_reg(X_train, X_test, y_train, y_test, args.solver)

    # Transpose the data in order to be computable by the network
    X_train = X_train.T
    X_test = X_test.T

    # Neural network
    layers = [32, 16]
    neuron = Neuron(layers=layers, nb_iter=150, learning_rate=0.1)
    parameters = neuron.fit(X_train, y_train, X_test, y_test)
    # y_pred = neuron.predict(X_test, parameters)
    y_pred = neuron.predict_softmax(X_test, parameters)
    print(y_pred)
    # diff = (y_pred == y_test).value_counts()
    # print(diff)

    # acc = accuracy_score(y_test, y_pred.T)
    # print(f'accuracy {acc:.2f}')
    # f1 = f1_score(y_true=y_test, y_pred=y_pred, pos_label='M')
    # print(f"f1_score: {f1:.2f}")

    # neuron.plot_cost_history(figsize=(15, 5), acc=True)

    plt.show()
