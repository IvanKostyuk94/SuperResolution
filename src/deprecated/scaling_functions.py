import numpy as np


def scale(data, epsilon=1e-9):
    return np.log(data + epsilon) / 25 + 1


def unscale(data, epsilon=1e-9):
    return np.exp(25 * (data - 1)) - epsilon


def scale_2(data, epsilon=1e-9):
    return np.log(data + epsilon) / 25


def unscale_2(data, epsilon=1e-9):
    return np.exp(25 * (data)) - epsilon


def scale_3(data, epsilon=1e-9):
    return np.log(data + epsilon)


def unscale_3(data, epsilon=1e-9):
    return np.exp(data) - epsilon


def scale_4(data, epsilon=1e-9):
    return np.log(data + epsilon) + 25


def unscale_4(data, epsilon=1e-9):
    return np.exp(data - 25) - epsilon
