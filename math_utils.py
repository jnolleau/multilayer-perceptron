from math import isnan, sqrt
import numpy as np


def count_column(serie):
    size = 0
    for val in serie:
        if (isnan(val) == False):
            size += 1
    return size


def mean_column(serie):
    sum = size = 0
    for val in serie:
        if (isnan(val) == False):
            sum += val
            size += 1
    return (sum / size, size) if size != 0 else (size, size)


def std_column(serie):
    sum = size = 0
    mean = mean_column(serie)[0]
    for val in serie:
        if (isnan(val) == False):
            sum += (val - mean) * (val - mean)
            size += 1
    return (sqrt(sum / size), size) if size != 0 else (size, size)


def min_max_column(serie):
    if serie.size == 0:
        return (0, 0)
    min = max = serie[0]
    for val in serie:
        if (isnan(val) == False):
            if min > val:
                min = val
            if max < val:
                max = val
    return (min, max)


def percentiles_column(serie):
    size = serie.size
    if size == 0:
        return (0, 0, 0)

    sorted_serie = np.sort(serie)
    q1 = (size + 3) // 4
    q2 = (size + 1) // 2
    q3 = (3 * size + 1) // 4
    return (sorted_serie[q1 - 1], sorted_serie[q2 - 1], sorted_serie[q3 - 1])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
