#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# In[2]:


model_retrain = True


# In[ ]:


# In[ ]:


# # Diffusion model

# In[3]:


# Imports:

# defaults
import csv
import functools
import itertools
import math
from fractions import Fraction

# Typenotes
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas

# from functools import partial
# from einops import rearrange, reduce
# from einops.layers.torch import Rearrange
# Smact check dependencies
import smact

# torch stuff
import torch
import torch.nn as nn
from dataset_creation import *

# UNet dependencies
from denoising_diffusion_pytorch import Unet1D  # fancy unet
from helper_dataset_shuffle import *

# file import
from helper_formula_parse import *
from helper_reverse_formula import *
from pymatgen.core.composition import Composition
from save_valid_compounds_to_csv import *
from smact.screening import pauling_test

# from helper_unet_functions import *
from smact_validity_checks import *
from supercon_wtypes_parse import *
from torch.optim import Adam, NAdam
from torch.utils.data import DataLoader, TensorDataset

# In[ ]:


# In[4]:


# データセット関連のハイパラ
NUM_TYPE_OF_ELEMENTS = 86  # 96 or 86
TYPE_OF_DATA = "unconditional"  # "unconditional" or "cuprate" or "pnictide" or "others"
SUPERCON_DATA_FILE = "./SuperCon_with_types.dat"


def create_dataset(NUM_TYPE_OF_ELEMENTS, TYPE_OF_DATA, SUPERCON_DATA_FILE):
    assert TYPE_OF_DATA in ["unconditional", "cuprate", "pnictide", "others"]
    assert NUM_TYPE_OF_ELEMENTS in [96, 86]

    # element table to set up vectors in R^(1x96): must len(element_table) = 96
    element_table = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
    ]

    # validate table correctness
    validation_element_table = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
    ]

    element_table = element_table[:NUM_TYPE_OF_ELEMENTS]
    validation_element_table = validation_element_table[:NUM_TYPE_OF_ELEMENTS]

    assert len(element_table) == NUM_TYPE_OF_ELEMENTS
    print("NOTE: Correct element table length.")

    assert validation_element_table == element_table
    print("NOTE: Valid table.")
    returned_datasets = prepare_datasets_for_classes(
        SUPERCON_DATA_FILE, element_table, 1 / 20, True
    )

    # Unpack the list into individual variables (if needed)
    torch_diffusion_data_raw_unconditional_train = returned_datasets[0]
    torch_diffusion_data_raw_cuprates_train = returned_datasets[1]
    torch_diffusion_data_raw_pnictides_train = returned_datasets[2]
    torch_diffusion_data_raw_others_train = returned_datasets[3]

    torch_diffusion_data_raw_unconditional_test = returned_datasets[4]
    torch_diffusion_data_raw_cuprates_test = returned_datasets[5]
    torch_diffusion_data_raw_pnictides_test = returned_datasets[6]
    torch_diffusion_data_raw_others_test = returned_datasets[7]

    # Select the data to use
    if TYPE_OF_DATA == "unconditional":
        torch_diffusion_data_raw_train = torch_diffusion_data_raw_unconditional_train
        torch_diffusion_data_raw_test = torch_diffusion_data_raw_unconditional_test
    elif TYPE_OF_DATA == "cuprate":
        torch_diffusion_data_raw_train = torch_diffusion_data_raw_cuprates_train
        torch_diffusion_data_raw_test = torch_diffusion_data_raw_cuprates_test
    elif TYPE_OF_DATA == "pnictide":
        torch_diffusion_data_raw_train = torch_diffusion_data_raw_pnictides_train
        torch_diffusion_data_raw_test = torch_diffusion_data_raw_pnictides_test
    elif TYPE_OF_DATA == "others":
        torch_diffusion_data_raw_train = torch_diffusion_data_raw_others_train
        torch_diffusion_data_raw_test = torch_diffusion_data_raw_others_test

    # 仕様する元素の制限
    if NUM_TYPE_OF_ELEMENTS == 86:
        # 86以降の元素がないデータを選ぶ
        num_original_train = len(torch_diffusion_data_raw_train)
        num_original_test = len(torch_diffusion_data_raw_test)
        torch_diffusion_data_raw_train = torch_diffusion_data_raw_train[
            torch_diffusion_data_raw_train[:, 86:].sum(dim=1) < 1e-7
        ]
        # torch_diffusion_data_raw_train = torch_diffusion_data_raw_train[:, :86]
        torch_diffusion_data_raw_test = torch_diffusion_data_raw_test[
            torch_diffusion_data_raw_test[:, 86:].sum(dim=1) < 1e-7
        ]
        # torch_diffusion_data_raw_test = torch_diffusion_data_raw_test[:, :86]
        num_processed_train = len(torch_diffusion_data_raw_train)
        num_processed_test = len(torch_diffusion_data_raw_test)
        print(
            f"NUM_TYPE_OF_ELEMENTS:{NUM_TYPE_OF_ELEMENTS}. Num train data {num_original_train} -> {num_processed_train}. Num test data {num_original_test} -> {num_processed_test}"
        )

    print(
        "torch_diffusion_data_raw_train.shape: ", torch_diffusion_data_raw_train.shape
    )
    print("torch_diffusion_data_raw_test.shape: ", torch_diffusion_data_raw_test.shape)
    return (
        torch_diffusion_data_raw_train,
        torch_diffusion_data_raw_test,
        element_table,
        validation_element_table,
    )


# In[5]:


def get_named_beta_schedule(
    schedule_name: str, num_diffusion_timesteps: int
) -> torch.Tensor:
    """
    Get a pre-defined beta schedule for the given name.
    Function adapted from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    Improved support for PyTorch.

    :param schedule_name: The name of the beta schedule.
    :param num_diffusion_timesteps: The number of diffusion timesteps.
    :return: The beta schedule tensor.
    :rtype: torch.Tensor[torch.float64]
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps).to(
            torch.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


# In[6]:


def betas_for_alpha_bar(
    num_diffusion_timesteps: int, alpha_bar: float, max_beta=0.999
) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumprod of (1-beta) over time from t = [0,1].

    Function adapted from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    Improved support for PyTorch.

    :param num_diffusion_timesteps: The number of betas to produce.
    :param alpha_bar: A lambda that takes an argument t from 0 to 1 and produces
                      the cumulative product of (1-beta) up to that part of the
                      diffusion process.
    :param max_beta: The maximum beta to use; use values lower than 1 to prevent
                     singularities (Improved Diffusion Paper).
    :return: The beta schedule tensor.
    :rtype: torch.Tensor[torch.float64]
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.Tensor(betas).to(torch.float64)


# In[7]:


class GaussianDiffusion1D:
    """
    Class for Gaussian diffusion of 1D Tensors (vector diffusion).
    Standard pixel-space DDPM diffusion process, extended to work for 1d tensors instead of 2d tensors.
    """

    def __init__(self, sequence_length: int, timesteps: int, beta_schedule_type: str):
        """
        Initializes the GaussianDiffusion1D class.

        :param sequence_length: Length of the sequence.
        :param timesteps: Number of timesteps.
        :param beta_schedule_type: Type of beta schedule. Can be "linear" or "cosine".

        :raises TypeError: If the beta schedule type is unknown.
        """
        self.sequence_length = sequence_length
        self.timesteps = timesteps
        self.beta_schedule_type = beta_schedule_type

        if self.beta_schedule_type == "linear":
            self.betas = get_named_beta_schedule(
                self.beta_schedule_type, self.timesteps
            )
        elif self.beta_schedule_type == "cosine":
            self.betas = get_named_beta_schedule(
                self.beta_schedule_type, self.timesteps
            )
        else:
            raise TypeError(
                f"{self.beta_schedule_type} is an unknown beta schedule type."
            )

        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)

    def forward(
        self, x_0: torch.Tensor, t: torch.Tensor, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process. Adding noise ~ N(0, I) to vectors.

        :param x_0: Original vector of shape (B, C, L).
        :param t: Timestep tensor of shape (B,).
        :param device: Device to be used.

        :return: Tuple containing mean tensor and noise tensor.
        """
        epsilon = torch.randn_like(x_0)
        alphas_bar_t = self.extract(self.alphas_bar, t, x_0.shape)

        mean = torch.sqrt(alphas_bar_t).to(device) * x_0.to(device)
        variance = torch.sqrt((1 - alphas_bar_t)).to(device) * epsilon.to(device)

        return mean + variance, epsilon.to(device)

    @torch.no_grad()
    def backward(
        self, x_t: torch.Tensor, t: torch.Tensor, model: nn.Module, **kwargs
    ) -> torch.Tensor:
        """
        Calls the model to predict the noise in the image and returns
        the denoised image (x_{t-1}).

        This method corresponds to the "big for loop" in the sampling algorithm (see algorithm 2 from Ho et al.).

        :param x_t: Current image tensor of shape (B, C, L).
        :param t: Timestep tensor of shape (1,).
        :param model: Model used to predict the noise in the image.
        :param **kwargs: Additional arguments to be passed to the model.

        :return: Denoised image tensor of shape (B, C, L).
        """
        betas_t = self.extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_bar_t = self.extract(
            torch.sqrt(1.0 - self.alphas_bar), t, x_t.shape
        )
        sqrt_recip_alphas_t = self.extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)
        mean = sqrt_recip_alphas_t * (
            x_t - ((betas_t / sqrt_one_minus_alphas_bar_t) * model(x_t, t, **kwargs))
        )
        posterior_variance_t = betas_t

        # Applies noise to this image if we are not in the last step yet.
        if t == 0:
            return mean
        else:
            z = torch.randn_like(x_t)
            variance = torch.sqrt(posterior_variance_t) * z
            return mean + variance

    @staticmethod
    def extract(
        values: torch.Tensor, t: torch.Tensor, x_0_shape: Tuple[int]
    ) -> torch.Tensor:
        """
        Picks the values from `values` according to the indices stored in `t`.

        :param values: Tensor of values to pick from.
        :param t: Index tensor.
        :param x_0_shape: Shape of the original tensor x_0.

        :return: Reshaped tensor with picked values.
        """
        batch_size = t.shape[0]
        vector_to_reshape = values.gather(-1, t.cpu())
        """
        if len(x_shape) - 1 = 2:
        reshape `out` to dims
        (batch_size, 1, 1)
        """
        return vector_to_reshape.reshape(batch_size, *((1,) * (len(x_0_shape) - 1))).to(
            t.device
        )


# In[8]:


def plot_noise_distribution(noise: torch.Tensor, predicted_noise: torch.Tensor):
    """
    Plot noise distributions to visualize and compare predicted and ground truth noise.
    """
    plt.hist(
        noise.cpu().numpy().flatten(),
        density=True,
        alpha=0.8,
        label="ground truth noise",
    )
    plt.hist(
        predicted_noise.cpu().numpy().flatten(),
        density=True,
        alpha=0.8,
        label="predicted noise",
    )
    plt.legend()
    plt.show()


# In[9]:


DIFFUSION_TIMESTEPS = 1000
diffusion_model = GaussianDiffusion1D(96, DIFFUSION_TIMESTEPS, "cosine")


# In[10]:


# Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 24  # TODO: test different values - change much bigger actually uses CPU (change to like 12)
NO_EPOCHS = 100
PRINT_FREQUENCY = 10
LR = 1e-4
VERBOSE = False
USE_VALIDATION_SET = True


# In[11]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(
    f'NOTE: Using Device: "{device}"',
    "|",
    (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"),
)


# # 学習と逆拡散過程によるサンプリングのコード

# In[12]:


import os
from typing import Optional


def train_diffusion_model(
    NUM_TYPE_OF_ELEMENTS: int,
    TYPE_OF_DATA: str,
    model_retrain: bool,
    LR: float,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: torch.device,
    fine_tune: bool = False,
    unet: Optional[nn.Module] = None,
):
    if fine_tune:
        assert unet is not None
        best_model_path = f"./models/best_unet_param_ELEM-{NUM_TYPE_OF_ELEMENTS}_SC-{TYPE_OF_DATA}_FT.pth"
    else:
        unet = Unet1D(dim=48, dim_mults=(1, 2, 3, 6), channels=1)
        unet.to(device)
        best_model_path = f"./models/best_unet_param_ELEM-{NUM_TYPE_OF_ELEMENTS}_SC-{TYPE_OF_DATA}.pth"

    if (not model_retrain) and os.path.exists(best_model_path):
        return best_model_path

    best_validation_loss = 1e9
    optimizer = torch.optim.NAdam(unet.parameters(), lr=LR)
    # Training Loop
    training_steps_tracker_train = []
    training_steps_tracker_val = []
    loss_tracker_train = []  # to plot train loss/training step - make sure outside for loop
    loss_tracker_val = []  # to plot val loss/training step - make sure outside for loop
    epoch_tracker = []
    loss_tracker_train_epoch = []  # to plot train loss/epoch - make sure outside for loop
    loss_tracker_val_epoch = []  # to plot val loss/epoch - make sure outside for loop
    for epoch in range(NO_EPOCHS):
        mean_epoch_loss_train = []  # put in for loop - wipe clean each time
        mean_epoch_loss_val = []  # put in for loop - wipe clean each time

        # on train dataset
        for batch in train_dataloader:
            # sample t from uniform distribution of 1 to T
            t = (
                torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,))
                .long()
                .to(device)
            )
            batch_train = batch[0].unsqueeze(1).to(device)

            noisy_batch_train, gt_noise_train = diffusion_model.forward(
                batch_train, t, device
            )
            predicted_noise_train = unet(noisy_batch_train.to(torch.float32), t)

            optimizer.zero_grad()
            # loss(pred, target)
            loss = torch.nn.functional.mse_loss(predicted_noise_train, gt_noise_train)
            loss_tracker_train.append(loss.item())
            training_steps_tracker_train.append(1)
            mean_epoch_loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

        if USE_VALIDATION_SET == True:
            # on test dataset
            for batch in test_dataloader:
                t = (
                    torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,))
                    .long()
                    .to(device)
                )
                batch_val = batch[0].unsqueeze(1).to(device)

                noisy_batch_val, gt_noise_val = diffusion_model.forward(
                    batch_val, t, device
                )
                predicted_noise_val = unet(noisy_batch_val.to(torch.float32), t)

                loss = torch.nn.functional.mse_loss(predicted_noise_val, gt_noise_val)
                loss_tracker_train.append(loss.item())
                training_steps_tracker_val.append(1)
                mean_epoch_loss_val.append(loss.item())

        epoch_tracker.append(epoch)
        loss_tracker_train_epoch.append(np.mean(mean_epoch_loss_train))
        if USE_VALIDATION_SET == True:
            loss_tracker_val_epoch.append(np.mean(mean_epoch_loss_val))

        # print loss(s)
        if epoch == 0 or epoch % PRINT_FREQUENCY == 9:
            print("---")
            if USE_VALIDATION_SET == True:
                print(
                    f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss_train)} | Val Loss {np.mean(mean_epoch_loss_val)}"
                )
            else:
                print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss_train)}")
            if VERBOSE:
                with torch.no_grad():
                    plot_noise_distribution(gt_noise_train, predicted_noise_train)
                    if USE_VALIDATION_SET == True:
                        plot_noise_distribution(gt_noise_val, predicted_noise_val)

            if best_validation_loss > np.mean(mean_epoch_loss_val):
                best_validation_loss = np.mean(mean_epoch_loss_val)
                torch.save(unet.state_dict(), best_model_path)

    # Plot Loss(s) vs epochs

    # Plot and label the training and validation loss values
    plt.plot(epoch_tracker, loss_tracker_train_epoch, label="Training Loss")
    # plt.plot(epochs, loss_tracker_val, label='Validation Loss')
    if USE_VALIDATION_SET:
        plt.plot(epoch_tracker, loss_tracker_val_epoch, label="Validation Loss")

    # Add in a title and axes labels
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Set the tick locations
    # plt.xticks(arange(0, len(loss_tracker_train) + 1, 100))

    # # Display the plot
    plt.legend(loc="best")
    plt.show()

    return best_model_path


# In[13]:


import itertools

from IPython.display import clear_output

# In[14]:


# Scratch 学習
## ハイパーパラメータ
TYPE_OF_DATA_LIST = ["unconditional", "cuprate", "pnictide", "others"]
NUM_TYPE_OF_ELEMENTS_LIST = [96]
combinations = list(itertools.product(TYPE_OF_DATA_LIST, NUM_TYPE_OF_ELEMENTS_LIST))

for TYPE_OF_DATA, NUM_TYPE_OF_ELEMENTS in combinations:
    print("----------------------------------------")
    print(
        f"{combinations.index((TYPE_OF_DATA, NUM_TYPE_OF_ELEMENTS))}/{len(combinations)} training : {TYPE_OF_DATA} data with {NUM_TYPE_OF_ELEMENTS} elements"
    )
    print("----------------------------------------")
    ## データセットの構築
    (
        torch_diffusion_data_raw_train,
        torch_diffusion_data_raw_test,
        element_table,
        validation_element_table,
    ) = create_dataset(NUM_TYPE_OF_ELEMENTS, TYPE_OF_DATA, SUPERCON_DATA_FILE)
    diffusion_dataset_train = TensorDataset(torch_diffusion_data_raw_train)
    train_dataloader = DataLoader(
        diffusion_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    if USE_VALIDATION_SET == True:
        diffusion_dataset_test = TensorDataset(torch_diffusion_data_raw_test)
        test_dataloader = DataLoader(
            diffusion_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    ## モデルの学習
    best_model_path = train_diffusion_model(
        NUM_TYPE_OF_ELEMENTS,
        TYPE_OF_DATA,
        model_retrain,
        LR,
        train_dataloader,
        test_dataloader,
        device,
    )
    print("\n\n\n\n")


# In[22]:


# Fine-tuning 学習
TYPE_OF_DATA_SCRATCH = (
    "unconditional"  # "unconditional" or "cuprate" or "pnictide" or "others"
)
NUM_TYPE_OF_ELEMENTS = 86  # 96 or 86
TYPE_OF_DATA = "unconditional"  # "unconditional" or "cuprate" or "pnictide" or "others"

TYPE_OF_DATA_LIST = ["cuprate", "pnictide", "others"]
NUM_TYPE_OF_ELEMENTS_LIST = [96]
combinations = list(itertools.product(TYPE_OF_DATA_LIST, NUM_TYPE_OF_ELEMENTS_LIST))

for TYPE_OF_DATA, NUM_TYPE_OF_ELEMENTS in combinations:
    print(
        f"{combinations.index((TYPE_OF_DATA, NUM_TYPE_OF_ELEMENTS))}/{len(combinations)} training : {TYPE_OF_DATA} data with {NUM_TYPE_OF_ELEMENTS} elements"
    )
    # 学習済みモデルの読み込み
    unet = Unet1D(dim=48, dim_mults=(1, 2, 3, 6), channels=1)
    unet.to(device)
    pretrained_model_path = train_diffusion_model(
        NUM_TYPE_OF_ELEMENTS,
        TYPE_OF_DATA_SCRATCH,
        model_retrain=False,
        LR=None,
        train_dataloader=None,
        test_dataloader=None,
        device=None,
    )
    unet.load_state_dict(torch.load(pretrained_model_path))

    # ファインチューニング用データセットの構築
    (
        torch_diffusion_data_raw_train,
        torch_diffusion_data_raw_test,
        element_table,
        validation_element_table,
    ) = create_dataset(NUM_TYPE_OF_ELEMENTS, TYPE_OF_DATA, SUPERCON_DATA_FILE)
    diffusion_dataset_train = TensorDataset(torch_diffusion_data_raw_train)
    train_dataloader = DataLoader(
        diffusion_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    if USE_VALIDATION_SET == True:
        diffusion_dataset_test = TensorDataset(torch_diffusion_data_raw_test)
        test_dataloader = DataLoader(
            diffusion_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    # ファインチューニング
    best_model_path = train_diffusion_model(
        NUM_TYPE_OF_ELEMENTS,
        TYPE_OF_DATA,
        model_retrain,
        LR,
        train_dataloader,
        test_dataloader,
        device,
        fine_tune=True,
        unet=unet,
    )


# In[ ]:


# In[ ]:


# In[ ]:
