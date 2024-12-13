import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import argparse
import sys
import mkl
import datetime

mkl.set_num_threads(1)

def load_data(file_path, train_fraction=0.75):
    """
    Load and prepare the data from a given npz file.

    Parameters
    ----------
    file_path : str
        Path to the .npz file containing 'wvl', 'labels', and 'flxn'.
    train_fraction : float
        Fraction of data to use for training. The rest is used for validation.

    Returns
    -------
    x : torch.FloatTensor
        Training input data.
    y : torch.FloatTensor
        Training target data.
    x_valid : torch.FloatTensor
        Validation input data.
    y_valid : torch.FloatTensor
        Validation target data.
    x_min : np.ndarray
        Minimum values used for input scaling.
    x_max : np.ndarray
        Maximum values used for input scaling.
    num_pix : int
        Number of output pixels.
    dim_in : int
        Dimensionality of input parameters.
    """

    temp = np.load(file_path)
    #wvl = temp["wvl"][::2]
    lbl = temp["labels"]
    flx = temp["flxn"][::2]

    # Example condition to filter data:
    new = lbl[0] > 0
    lbl = lbl[:, new]
    flx = flx[:, new]

    cvs = int(len(lbl[0]) * train_fraction)
    x_raw = lbl.T
    y_raw = flx.T

    x_train_raw = x_raw[:cvs, :]
    y_train_raw = y_raw[:cvs, :]

    x_valid_raw = x_raw[cvs:, :]
    y_valid_raw = y_raw[cvs:, :]

    # Scale the inputs
    x_max = np.max(x_train_raw, axis=0)
    x_min = np.min(x_train_raw, axis=0)

    x_train_scaled = (x_train_raw - x_min) / (x_max - x_min) - 0.5
    x_valid_scaled = (x_valid_raw - x_min) / (x_max - x_min) - 0.5

    x = Variable(torch.from_numpy(x_train_scaled)).float()
    y = Variable(torch.from_numpy(y_train_raw), requires_grad=False).float()
    x_valid = Variable(torch.from_numpy(x_valid_scaled)).float()
    y_valid = Variable(torch.from_numpy(y_valid_raw), requires_grad=False).float()

    dim_in = x.shape[1]
    num_pix = y.shape[1]

    return x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in


def build_model(dim_in, num_pix, hidden_neurons):
    """
    Build a simple feed-forward neural network with two hidden layers.

    Parameters
    ----------
    dim_in : int
        Dimensionality of input features.
    num_pix : int
        Number of output pixels (flux values).
    hidden_neurons : int
        Number of neurons in each hidden layer.

    Returns
    -------
    model : torch.nn.Sequential
        The defined neural network.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, hidden_neurons),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_neurons, hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_neurons, num_pix)
    )
    return model


def train_model(model, x, y, x_valid, y_valid,
                learning_rate=0.001, patience=20,
                check_interval=1000):
    """
    Train the model, performing early stopping based on validation loss.

    Parameters
    ----------
    model : torch.nn.Sequential
        The neural network model to train.
    x : torch.FloatTensor
        Training input data.
    y : torch.FloatTensor
        Training target data.
    x_valid : torch.FloatTensor
        Validation input data.
    y_valid : torch.FloatTensor
        Validation target data.
    learning_rate : float
        Learning rate for the optimizer.
    patience : int
        Number of times validation loss is allowed to not improve before stopping.
    check_interval : int
        Interval (in iterations) at which to check validation loss.

    Returns
    -------
    model_numpy : list of np.ndarray
        Best model parameters found during training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    current_loss = np.inf
    count = 0
    t = 0
    start_time = time.time()
    model_numpy = None

    while count < patience:
        y_pred = model(x)
        loss = ((y_pred - y).pow(2)).mean()

        # Check convergence
        if t % check_interval == 0:
            with torch.no_grad():
                y_pred_valid = model(x_valid)
                loss_valid = ((y_pred_valid - y_valid).pow(2)).mean()

            print(f"Iter: {t}, Patience Count: {count}, Valid Loss: {loss_valid}, Elapsed: {time.time() - start_time:.2f}s")

            if loss_valid > current_loss:
                count += 1
            else:
                current_loss = loss_valid
                # Record best parameters
                model_numpy = [param.data.numpy().copy() for param in model.parameters()]

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t += 1

    print(f"Training finished in {time.time() - start_time:.2f}s with final validation loss: {current_loss}")
    return model_numpy

def random_network_for_testing(model):
    """
    Randomize the model parameters for testing purposes.

    Parameters
    ----------
    model : torch.nn.Sequential
        The model to randomize.

    Returns
    -------
    model_numpy : list of np.ndarray
        Randomized model parameters.
    """
    model_numpy = [param.data.numpy().copy() for param in model.parameters()]
    for param in model_numpy:
        param[:] = np.random.randn(*param.shape)
    return model_numpy


def save_model_parameters(file_path, model_numpy, x_min, x_max):
    """
    Save the model parameters and scaling info to a .npz file.

    Parameters
    ----------
    file_path : str
        Path to the output file (without extension).
    model_numpy : list of np.ndarray
        Model parameters to save.
    x_min : np.ndarray
        Minimum values for input scaling.
    x_max : np.ndarray
        Maximum values for input scaling.
    """
    w_array_0 = model_numpy[0]
    b_array_0 = model_numpy[1]
    w_array_1 = model_numpy[2]
    b_array_1 = model_numpy[3]
    w_array_2 = model_numpy[4]
    b_array_2 = model_numpy[5]

    np.savez(file_path,
             w_array_0=w_array_0,
             w_array_1=w_array_1,
             w_array_2=w_array_2,
             b_array_0=b_array_0,
             b_array_1=b_array_1,
             b_array_2=b_array_2,
             x_max=x_max,
             x_min=x_min)


if __name__ == "__main__":
    learning_rate = 0.001
    patience = 20
    check_interval = 1000
    hidden_neurons = 300
    data_file = "/mnt/beegfs/gemini/groups/bergemann/users/shared-storage/kovalev/payne/mafs20-g1.npz"
    # today's date
    output_file = f"payne_ts_nlte_hr10_{datetime.datetime.now().strftime('%Y-%m-%d')}.npz"

    # Load the data
    x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in = load_data(data_file)

    # Build the model
    model = build_model(dim_in, num_pix, hidden_neurons)

    # Train the model
    model_numpy = train_model(model, x, y, x_valid, y_valid,
                              learning_rate=learning_rate,
                              patience=patience,
                              check_interval=check_interval)

    # Save parameters
    if model_numpy is not None:
        save_model_parameters(output_file, model_numpy, x_min, x_max)
        print(f"Model parameters saved to {output_file}.npz")
    else:
        print("No best model found, no file saved.")
