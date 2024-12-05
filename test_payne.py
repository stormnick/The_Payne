from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from The_Payne import utils
from The_Payne import spectral_model
from The_Payne import fitting
from The_Payne import training

matplotlib.use("MacOSX")

# Created by storm at 05.12.24

# load the default training set. Note that, due to the GitHub size limit,
# this training set is a small subset of what I used to train the default network
training_labels, training_spectra, validation_labels, validation_spectra = utils.load_training_data()
# label array unit = [n_spectra, n_labels]
# spectra_array unit = [n_spectra, n_pixels]

# The validation set is used to independently evaluate how well the neural net
# is emulating the spectra. If the network overfits the spectral variation, while
# the loss will continue to improve for the training set, the validation set
# should exhibit a worsen loss.

# the codes outputs a numpy array ""NN_normalized_spectra.npz"
# which stores the trained network parameters
# and can be used to substitute the default one in the directory neural_nets/
# it will also output a numpy array "training_loss.npz"
# which stores the progression of the training and validation losses

training.neural_net(training_labels, training_spectra,\
                    validation_labels, validation_spectra,\
                    num_neurons=300, learning_rate=1e-4,\
                    num_steps=1e4, batch_size=128)

# a larger batch_size (e.g. 512) when possible is desirable
# here we choose batch_size=128 above because the sample training set is limited in size

tmp = np.load("training_loss.npz") # the output array also stores the training and validation loss
training_loss = tmp["training_loss"]
validation_loss = tmp["validation_loss"]

plt.figure(figsize=(14, 4))
plt.plot(np.arange(training_loss.size)*100, training_loss, 'k', lw=0.5, label = 'Training set')
plt.plot(np.arange(training_loss.size)*100, validation_loss, 'r', lw=0.5, label = 'Validation set')
plt.legend(loc = 'best', frameon = False, fontsize= 18)
plt.yscale('log')
plt.ylim([5,100])
plt.xlabel("Step", size=20)
plt.ylabel("Loss", size=20)
plt.show()