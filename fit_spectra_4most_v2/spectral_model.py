# code for predicting the spectrum of a single star in normalized space.
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np

def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''
    return z*(z > 0) + 0.01*z*(z < 0)

def relu(z):
    return z*(z > 0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def get_spectrum_from_neural_net(scaled_labels, NN_coeffs,
                                 kovalev=False, kovalev_alt=False,
                                 pixel_limits=None):
    """
    Predict the rest-frame spectrum (normalized) of a single star.
    We input the scaled stellar labels (not in the original unit).
    Each label ranges from -0.5 to 0.5

    Parameters
    ----------
    scaled_labels : 1D array
        The scaled input labels to the model.
    NN_coeffs : tuple
        (w_array_0, w_array_1, w_array_2,
         b_array_0, b_array_1, b_array_2, x_min, x_max)
    kovalev, kovalev_alt : bool
        Flags to switch activation in the second hidden layer / output.
    pixel_limits : None, tuple, or list of tuples
        - None: compute the full output.
        - (start, end): compute pixels in [start:end).
        - [(start1, end1), (start2, end2), ... ]: multiple slices,
          each slice is concatenated into one final spectrum array.
    """

    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs

    # First hidden layer
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0

    # Activation depends on flags:
    if not kovalev and not kovalev_alt:
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        hidden_act = leaky_relu(outside)
        final_act = leaky_relu  # for consistency, though originally you do a direct linear
    elif kovalev:
        outside = np.einsum('ij,j->i', w_array_1, relu(inside)) + b_array_1
        hidden_act = sigmoid(outside)
        final_act = sigmoid
    else:  # kovalev_alt
        outside = np.einsum('ij,j->i', w_array_1, relu(inside)) + b_array_1
        hidden_act = relu(outside)
        final_act = sigmoid

    # If pixel_limits is None -> compute all pixels
    if pixel_limits is None:
        # Full final layer
        spectrum = np.einsum('ij,j->i', w_array_2, hidden_act) + b_array_2
        spectrum = final_act(spectrum)
        return spectrum

    # If pixel_limits is a single (start, end) pair -> compute that window
    # If pixel_limits is a list of (start, end) pairs -> compute them all and concatenate
    if isinstance(pixel_limits[0], (list, tuple)):
        # Multiple slices
        partial_spectra = []
        for (start, end) in pixel_limits:
            w_sub = w_array_2[start:end]
            b_sub = b_array_2[start:end]
            partial = np.einsum('ij,j->i', w_sub, hidden_act) + b_sub
            partial = final_act(partial)
            partial_spectra.append(partial)
        # Concatenate into one final array
        spectrum = np.concatenate(partial_spectra, axis=0)
    elif len(pixel_limits) == 2:
        # Single slice
        start, end = pixel_limits
        w_sub = w_array_2[start:end]
        b_sub = b_array_2[start:end]
        spectrum = np.einsum('ij,j->i', w_sub, hidden_act) + b_sub
        spectrum = final_act(spectrum)
    elif len(pixel_limits) == len(w_array_2):
        # array of True/False probably
        w_sub = w_array_2[pixel_limits]
        b_sub = b_array_2[pixel_limits]
        spectrum = np.einsum('ij,j->i', w_sub, hidden_act) + b_sub
        spectrum = final_act(spectrum)
    else:
        raise ValueError("pixel_limits must be None, a tuple, or a list of tuples.")

    return spectrum