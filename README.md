# The Payne
Tools for interpolating spectral models with neural networks.

## Installation
Clone this repository and run code from the base directory.
```
python setup.py install
````

The [tutorial](https://github.com/tingyuansen/The_Payne/blob/master/tutorial.ipynb) shows some simple use cases.

## Dependencies
* The spectral model and fitting routines require only Numpy and Scipy.
* Training a new neural network requires [PyTorch](http://pytorch.org/) (GPUs required).
* All these dependencies will be automatically installed alongside with this package
* I develop this package in Python 3.7 using Anaconda.

## Citing this code
* Please cite [Ting et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T/abstract), when using this code. The paper describes the method and its application to APOGEE spectra.

## Authors
* [Yuan-Sen Ting](http://www.sns.ias.edu/~ting/) -- ting at ias dot edu
* [Kareem El-Badry](http://w.astro.berkeley.edu/~kelbadry/)

## Licensing

Copyright 2019 by Yuan-Sen Ting.

This software is governed by the MIT License: In brief, you can use, distribute, and change this package as you please.


## added

1) Computing synthetic spectra
This should be done entirely on your side. Depending on the number of parameters, you’ll need anywhere from a few thousand to a few tens of thousands of spectra. Just make sure there are no problematic spectra (e.g. those containing NaNs, negative fluxes, etc.), as those initially prevented my network from converging properly.

I trained on fully normalised spectra, and my current network constrains the output between 0 and 1. Feel free to change the architecture if that assumption doesn’t apply to your case.

2) Reformatting the spectra into one large .npz file
This simplifies loading for training. I use the script prep_data_no_vmac_all_labels_hr10.py for this. It:

-Loads the spectra
-Cuts them into my preferred wavelength ranges (line 18)
-Optionally interpolates the spectra to a larger Δλ (line 78; remove if using your input spectra as-is)
-Saves everything to an .npz file

Important details:

a) To get the labels for each spectrum, I use a .csv file. The first column must be named specname (line 20), matching the filenames of the synthetic spectra.
b) This same file is used to select parameter columns (line 22), and line 25 specifies which parameters to save for training.
c) Line 50: Teff is converted to kK.
d) Line 65: Spectra with NaNs or flux < 1e-8 are removed. Modify this line if that’s not required for your use case.
e) Line 72: Emission lines are set back to 1. Again, adjust as needed.
f) Line 34: Path to the directory with all the spectra. Line 52 loads spectra ending with .spec — change this if your file naming differs.
g) Line 37: Name of the .csv file with labels — change as needed.
h) Line 38: Name of the output .npz file — also update accordingly.

Tip: I recommend scaling your labels so they behave nicely during training. For instance, you might want to use A(Li) instead of [Li/Fe], etc.

I train on spectra without any vmac or vsini broadening. You can apply broadening here (e.g. see line 69), but be aware that it might reduce network performance.

3) Training the network
The script kovalev_network.py is used to start training. You can call it with the data file from step 2 (line 254) and the name of the network to save (line 255). Note that the network name will include a datetime string.

Things to note:

a) Lines 245–248 control the network parameters. These settings worked for spectra with 10k–30k wavelength points but may need increasing for larger spectra.
b) The function defining the architecture is in line 86. Modify this if you want to use a different architecture.
c) I trained on CPUs (unfortunately), so I’m not sure how much you’d need to adjust for GPU usage. I used 32 CPUs, and training usually took a few days.
d) The network prints the loss every 1000 steps. I typically reached convergence after 50–70k steps, with a final loss on the order of 1e-5 to 1e-6. Anything higher than that would be concerning.

4) Plotting the performance
This happens automatically within kovalev_network.py, which calls kovalev_plot_performance.py. It attempts to plot performance using the validation spectra (from the training data), and saves two plots — one coloured by [Fe/H].

This gives a rough idea of how the network is performing. If you changed the architecture, update the model in the function at line 117 of kovalev_plot_performance.py.