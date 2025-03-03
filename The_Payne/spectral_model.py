# code for predicting the spectrum of a single star in normalized space.
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from . import utils

def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''
    return z*(z > 0) + 0.01*z*(z < 0)

def relu(z):
    return z*(z > 0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def get_spectrum_from_neural_net(scaled_labels, NN_coeffs, kovalev=False, kovalev_alt=False):
    '''
    Predict the rest-frame spectrum (normalized) of a single star.
    We input the scaled stellar labels (not in the original unit).
    Each label ranges from -0.5 to 0.5
    '''

    # assuming your NN has two hidden layers.
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0

    if not kovalev and not kovalev_alt:
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
    elif kovalev:
        outside = np.einsum('ij,j->i', w_array_1, relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, sigmoid(outside)) + b_array_2
    elif kovalev_alt:
        outside = np.einsum('ij,j->i', w_array_1, relu(inside)) + b_array_1
        spectrum = sigmoid(np.einsum('ij,j->i', w_array_2, relu(outside)) + b_array_2)


    return spectrum
