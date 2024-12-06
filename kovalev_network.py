import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
#import mkl

#mkl.set_num_threads(1)

# choose a testing batch

name = "/mnt/beegfs/gemini/groups/bergemann/users/shared-storage/kovalev/payne/mafs20-g1.npz"
temp = np.load(name)

wvl = temp["wvl"][::2]
lbl = temp["labels"]  # [:,:2950]
new = lbl[0] > 0
flx = temp["flxn"][::2, new]
lbl = temp["labels"][:, new]
nlsel = np.arange(wvl.shape[0])
np.set_printoptions(suppress=True)
ins = []
plt.scatter(lbl[0], lbl[1], c=lbl[2])
plt.colorbar()
plt.show()
cvs = int(len(lbl[0]) * 0.75)
print("CVS=", lbl.shape, cvs)

ww = np.ones(lbl.shape[1])
wv = ww[cvs:]
ww = ww[:cvs]
wv = Variable(torch.from_numpy(wv), requires_grad=False).type(torch.FloatTensor)
ww = Variable(torch.from_numpy(ww), requires_grad=False).type(torch.FloatTensor)

x = (lbl.T)[:cvs, :]
y = (flx.T)[:cvs, :]  # num_go*570:(num_go+1)*570]

# and validation spectra
x_valid = (lbl.T)[cvs:, :]
y_valid = (flx.T)[cvs:, :]  # num_go*570:(num_go+1)*570]#num_go*150:(num_go+1)*150]

# scale the labels
x_max = np.max(x, axis=0)
x_min = np.min(x, axis=0)

x = (x - x_min) / (x_max - x_min) - 0.5
x_valid = (x_valid - x_min) / (x_max - x_min) - 0.5

# dimension of the input
dim_in = x.shape[1]
num_pix = y.shape[1]

# make pytorch variables
x = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
y = Variable(torch.from_numpy(y), requires_grad=False).type(torch.FloatTensor)
x_valid = Variable(torch.from_numpy(x_valid)).type(torch.FloatTensor)
y_valid = Variable(torch.from_numpy(y_valid), \
                   requires_grad=False).type(torch.FloatTensor)

# =============================================================================
# define neural network
model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, 300),
    torch.nn.ReLU(),
    torch.nn.Linear(300, 300),
    torch.nn.Sigmoid(),
    torch.nn.Linear(300, num_pix)
)

# define optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# convergence counter
current_loss = np.inf
count = 0
t = 0

# record time
start_time = time.time()

# train the neural network
while count < 20:

    # training
    y_pred = model(x)
    loss = ((y_pred - y).pow(2)).mean()

    # check convergence
    if t % 1000 == 0:
        y_pred_valid = model(x_valid)
        loss_valid = ((y_pred_valid - y_valid).pow(2)).mean()
        print(count, loss_valid, time.time() - start_time)

        if loss_valid > current_loss:
            count += 1
        else:
            # record the best loss
            current_loss = loss_valid

            # record the best parameters
            model_numpy = []
            for param in model.parameters():
                model_numpy.append(param.data.numpy())

    # -----------------------------------------------------------------------------
    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t += 1

# print run time
print(time.time() - start_time)

# extract parameters
w_array_0 = model_numpy[0]
b_array_0 = model_numpy[1]
w_array_1 = model_numpy[2]
b_array_1 = model_numpy[3]
w_array_2 = model_numpy[4]
b_array_2 = model_numpy[5]

# save parameters and remember how we scale the labels
np.savez("NN_results_RrelsigN" + str(20) + ".npz", \
         w_array_0=w_array_0, \
         w_array_1=w_array_1, \
         w_array_2=w_array_2, \
         b_array_0=b_array_0, \
         b_array_1=b_array_1, \
         b_array_2=b_array_2, \
         x_max=x_max, \
         x_min=x_min)
