import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import time
import argparse
import sys
#import mkl
import datetime
from sys import argv
from kovalev_plot_performance import do_plot
import os, math, pathlib
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler          # AMP = mixed-precision
from torch.cuda import autocast          # AMP = mixed-precision

#mkl.set_num_threads(1)

def load_and_stack(file_paths):
    """
    Load one or more NPZ spectral files and concatenate their spectra.

    Parameters
    ----------
    file_paths : sequence of str or Path
        Paths to .npz files that each contain the keys
        'wvl', 'labels', 'flxn', and 'label_names'.

    Returns
    -------
    wvl : ndarray
        Shared wavelength grid (1-D).
    lbl : ndarray
        Labels stacked column-wise, shape (n_labels, total_spectra).
    flx : ndarray
        Fluxes stacked column-wise, shape (n_pixels, total_spectra).
    label_names : ndarray
        The label names array (1-D).
    """
    # Accept a single path as well as a list/tuple
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    lbl_blocks, flx_blocks = [], []
    wvl = label_names = None

    for idx, fp in enumerate(file_paths, 1):
        with np.load(fp) as f:
            # --- consistency checks (first file defines the reference) ----
            if idx == 1:
                wvl = f["wvl"]
                label_names = f["label_names"]
            else:
                if not np.allclose(f["wvl"], wvl):
                    raise ValueError(f"{fp} has a different wavelength grid.")
                if not np.array_equal(f["label_names"], label_names):
                    raise ValueError(f"{fp} has different label_names ordering.")

            # --- collect blocks ------------------------------------------
            lbl_blocks.append(f["labels"])
            flx_blocks.append(f["flxn"])

    # --- final concatenation --------------------------------------------
    lbl = np.concatenate(lbl_blocks, axis=1)
    flx = np.concatenate(flx_blocks, axis=1)

    return wvl, lbl, flx, label_names

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

    if type(file_path) is list:
        # If multiple files are provided, stack them
        wvl, lbl, flx, label_names = load_and_stack(file_path)
    else:
        temp = np.load(file_path)
        wvl = temp["wvl"]
        lbl = temp["labels"]
        flx = temp["flxn"]
        label_names = temp["label_names"]
        temp.close()

    # Example condition to filter data:
    #new = lbl[0] > 0
    #lbl = lbl[:, new]
    #flx = flx[:, new]

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

    return x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in, wvl, label_names


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
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_neurons, num_pix),
        torch.nn.Sigmoid()  # enforce [0, 1] outputs
    )
    return model, "Linear-ReLU-Linear-ReLU-Linear-Sigmoid"


def train_model(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        *,
        batch_size      : int   = 4096,
        valid_batch_size: int   = 8192,
        base_lr         : float = 1e-3,
        weight_decay    : float = 1e-2,
        t_max           : int|None = None,          # steps per cosine period
        patience        : int   = 20,
        check_interval  : int   = 1_000,            # validate & maybe checkpoint
        checkpoint_dir  : str   = "checkpoints",
        amp             : bool  = True,             # turn mixed precision on/off
        device          : str|None = None
    ):
    """
    Memory-safe GPU trainer with AdamW, cosine LR, AMP and robust checkpointing.
    """

    # ───── DEVICE ─────────────────────────────────────────
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)              # single-node multi-GPU
    # For multi-node use torchrun + DistributedDataParallel.

    # ───── DATA LOADERS ──────────────────────────────────
    train_dl = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    valid_dl = DataLoader(
        TensorDataset(x_valid, y_valid),
        batch_size=valid_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )

    # ───── OPTIMISER & SCHEDULER ─────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )

    # If user did not specify T_max, pick “~50 passes over train_dl”
    if t_max is None:
        t_max = 50 * len(train_dl)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=0.0
    )

    # ───── AMP SETUP ─────────────────────────────────────
    scaler = GradScaler(enabled=amp)

    # ───── BOOK-KEEPING ─────────────────────────────────
    ckpt_root     = pathlib.Path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    impatience    = 0
    step          = 0
    start         = time.perf_counter()

    # ───── TRAIN LOOP ───────────────────────────────────
    while impatience < patience:
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=amp):
                pred  = model(xb)
                loss  = (pred - yb).pow(2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ───── VALIDATE / CHECKPOINT ────────────────
            if step % check_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad(), autocast('cuda', enabled=amp):
                    for xb_v, yb_v in valid_dl:
                        xb_v, yb_v = xb_v.to(device, non_blocking=True), yb_v.to(device, non_blocking=True)
                        val_losses.append(((model(xb_v) - yb_v).pow(2)).mean().item())
                val_loss = float(np.mean(val_losses))

                hr = (time.perf_counter() - start) / 3600
                print(f"[{step:>7}]  val_loss={val_loss:.4e} | lr={scheduler.get_last_lr()[0]:.2e} | elapsed={hr:.2f} h")

                if val_loss < best_val_loss:          # ▸ improved
                    best_val_loss = val_loss
                    impatience    = 0

                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = ckpt_root / f"step{step:07d}_val{val_loss:.4e}_{ts}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "scaler_state":   scaler.state_dict(),
                            "val_loss": val_loss
                        },
                        fname
                    )
                    print(f"    ★  saved {fname}")
                else:
                    impatience += 1

            step += 1
            if impatience >= patience:
                break   # early-stop exits inner loop as well

    # ───── WRAP-UP ───────────────────────────────────────
    hours = (time.perf_counter() - start) / 3600
    print(f"\nStopped after {step} steps ({hours:.2f} h). Best val_loss = {best_val_loss:.5e}")

    # Reload best weights, move to CPU, convert to numpy list
    best_ckpt = sorted(ckpt_root.glob("*.pt"))[-1]
    state = torch.load(best_ckpt, map_location="cpu")["model_state"]

    model_cpu = (model.module if isinstance(model, nn.DataParallel) else model).cpu()
    model_cpu.load_state_dict(state)

    model_numpy = [p.detach().numpy().copy() for p in model_cpu.parameters()]

    meta = dict(
        base_lr        = base_lr,
        weight_decay   = weight_decay,
        batch_size     = batch_size,
        amp            = amp,
        t_max          = t_max,
        patience       = patience,
        check_interval = check_interval,
        best_val_loss  = best_val_loss,
        total_steps    = step,
        device         = device,
        checkpoint_dir = str(ckpt_root.resolve())
    )
    return model_numpy, meta

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


def save_model_parameters(file_path, model_numpy, x_min, x_max, meta_data):
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
             x_min=x_min,
                **meta_data)


if __name__ == "__main__":
    learning_rate = 0.001  # original 0.001
    patience = 20
    check_interval = 1000
    hidden_neurons = 300
    weight_decay = 0.001

    #data_file = "/mnt/beegfs/gemini/groups/bergemann/users/storm/payne/feb2025/grid_nlte_goodwavelength_allnlteelements_again_ts_batch0_hr10_novmac_alt_inf_res_feb2025.npz"
    #data_file = "/mnt/beegfs/gemini/groups/bergemann/users/shared-storage/kovalev/payne/mafs20-g1.npz"
    # today's date
    #output_file = f"payne_ts_nlte_allnlteelements_again_hr10_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.npz"
    data_file = argv[1]
    output_file = f"{argv[2]}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.npz"

    # print current time
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Training network with {hidden_neurons} hidden neurons, learning rate {learning_rate} and patience {patience} using data from {data_file} and saving to {output_file}")
    # Load the data
    x, y, x_valid, y_valid, x_min, x_max, num_pix, dim_in, wvl, label_names = load_data(data_file)

    # Build the model
    model, model_architecture = build_model(dim_in, num_pix, hidden_neurons)

    # Train the model
    model_numpy, meta_data = train_model(model, x, y, x_valid, y_valid,
                              base_lr=learning_rate,
                              patience=patience,
                              check_interval=check_interval, weight_decay=weight_decay)

    meta_data["date"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta_data["wavelength"] = wvl
    meta_data["label_names"] = label_names
    meta_data["author"] = 'storm'
    meta_data['model_architecture'] = model_architecture
    meta_data['data_file_path'] = data_file

    # Save parameters
    if model_numpy is not None:
        save_model_parameters(output_file, model_numpy, x_min, x_max, meta_data)
        print(f"Model parameters saved to {output_file}")
    else:
        print("No best model found, no file saved.")

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    do_plot(data_file, output_file)