from __future__ import annotations

import numpy as np

from The_Payne import utils
from The_Payne import spectral_model
from The_Payne import fitting
import time
import timeit

# Created by storm at 09.12.24

# if you trained your own neural net (see last part of this tutorial),
# you can load in your own neural net
tmp = np.load("/Users/storm/PycharmProjects/payne/test_network/test_large1.npz")
w_array_0 = tmp["w_array_0"]
w_array_1 = tmp["w_array_1"]
w_array_2 = tmp["w_array_2"]
b_array_0 = tmp["b_array_0"]
b_array_1 = tmp["b_array_1"]
b_array_2 = tmp["b_array_2"]
x_min = tmp["x_min"]
x_max = tmp["x_max"]
tmp.close()
NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)

tmp = np.load("/Users/storm/PycharmProjects/payne/test_network/NN_results_RrelsigN20_sapp.npz")
w_array_0 = tmp["w_array_0"]
w_array_1 = tmp["w_array_1"]
w_array_2 = tmp["w_array_2"]
b_array_0 = tmp["b_array_0"]
b_array_1 = tmp["b_array_1"]
b_array_2 = tmp["b_array_2"]
x_min = tmp["x_min"]
x_max = tmp["x_max"]
tmp.close()
NN_coeffs_sapp = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)

real_labels = [5.77, 4.44, 0.0, 1.0, 4., 0., 0., 0]
scaled_labels = (real_labels-x_min)/(x_max-x_min) - 0.5

# real_spec = spectral_model.get_spectrum_from_neural_net(scaled_labels = scaled_labels, NN_coeffs = NN_coeffs, kovalev=True)
# real_spec_sapp = spectral_model.get_spectrum_from_neural_net(scaled_labels = scaled_labels, NN_coeffs = NN_coeffs_sapp, kovalev=True)

# test timing
# Number of iterations for timing
num_iterations = 1000

# Timing the function multiple times
timings = timeit.repeat(stmt='spectral_model.get_spectrum_from_neural_net(scaled_labels = scaled_labels, NN_coeffs = NN_coeffs, kovalev=True)',
                        setup='from The_Payne import spectral_model',
                        repeat=3,
                        number=num_iterations,
                        globals=globals())

# Convert timings to seconds (if needed)
timings = np.array(timings) / num_iterations * 1000

# Calculate statistics
average_time = np.mean(timings)
std_deviation = np.std(timings)

print(f"Average execution time my Payne: {average_time:.6f} +/- {std_deviation:.6f} ms")

# Timing the function multiple times
timings = timeit.repeat(stmt='spectral_model.get_spectrum_from_neural_net(scaled_labels = scaled_labels, NN_coeffs = NN_coeffs_sapp, kovalev=True)',
                        setup='from The_Payne import spectral_model',
                        repeat=3,
                        number=num_iterations,
                        globals=globals())

# Convert timings to seconds (if needed)
timings = np.array(timings) / num_iterations * 1000

# Calculate statistics
average_time = np.mean(timings)
std_deviation = np.std(timings)

print(f"Average execution time SAPP: {average_time:.6f} +/- {std_deviation:.6f} ms")