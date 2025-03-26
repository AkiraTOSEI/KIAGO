import itertools
from itertools import combinations
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def nearest_multiple_torch(x: torch.Tensor, y: float) -> torch.Tensor:
    """Find the nearest multiple of y to x using PyTorch.

    Args:
        x (torch.Tensor): The number for which we want to find the nearest multiple.
        y (float): The base multiple.

    Returns:
        torch.Tensor: The nearest multiple of y to x.
    """
    return torch.round(x / y) * y


def force_atoms_to_crystal_structure_loss(
    atoms: torch.Tensor, structure_atom_counts: torch.Tensor
) -> torch.Tensor:
    """Compute the loss for aligning atom distributions to a specified crystal structure atom count.

    The function computes the loss based on the distance of each atom value
    to its nearest multiple of 1/structure_atom_count, encouraging the alignment of atom distributions
    to the specified crystal structure atom count.

    The objective of this loss function is to ensure each element in the 'atoms' tensor aligns
    with the inverse of the specified atom count in the crystal structure. Ideally, each atom
    value should be a positive integer multiple of the inverse of the structure's atom count.
    For instance, with a structure_atom_count of 10 and atoms tensor as [0.0, 0.07, 0.21, 0.32, 0.4, ...],
    each atom value should ideally be among [0.0, 0.1, 0.2, 0.3, 0.4, ..., 1.0]. The loss is computed
    based on the squared distance between each atom value and its nearest ideal value. For example,
    0.07 should ideally be 0.1, so its loss contribution would be (0.07 - 0.1)^2 = 0.03^2.


    Args:
        atoms (torch.Tensor): The 2D probability distribution.
        structure_atom_count (int): The specified atom count for the crystal structure.

    Returns:
        torch.Tensor: The computed loss for alignment.
    """
    inv_structure_atom_counts = (1 / structure_atom_counts).unsqueeze(1).unsqueeze(2)
    nearest_multiples = (
        torch.round(atoms.unsqueeze(0) / inv_structure_atom_counts)
        * inv_structure_atom_counts
    )
    losses = (atoms.unsqueeze(0) - nearest_multiples) ** 2
    return losses  # Shape: (len(structure_atom_counts), batch_size, num_atoms)


class IntegerLoss(nn.Module):
    def __init__(self, structure_atom_count_range=range(4, 21), num_types_of_atoms=8):
        """
        Initialize the optimal structure loss calculator.

        Args:
            structure_atom_count_range (range): Range of atom counts for different crystal structures.
            num_types_of_atoms (int): Total number of types of atoms.
        """
        super().__init__()
        self.structure_atom_counts = nn.Parameter(
            torch.tensor(list(structure_atom_count_range), dtype=torch.float32),
            requires_grad=False,
        )
        self.num_types_of_atoms = num_types_of_atoms
        self.dummy_mask = np.ones((1, num_types_of_atoms)).astype(np.float32)
        self._create_ids()

    def _create_ids(self):
        self.structure_ids = np.array(list(range(len(self.structure_atom_counts))))

    def calculate_min_loss(
        self, atoms: torch.Tensor, reduction: str = "none"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the minimum loss for aligning atom distributions over a range of crystal structure atom counts using PyTorch.

        This method computes the loss for each structure atom count in the provided range using the
        force_atoms_to_crystal_structure_loss function, and returns the minimum loss value, the corresponding index,
        and the used mask.

        Args:
            atoms (torch.Tensor): The 2D probability distribution.
            reduction (str): Specifies the reduction to apply to the output. Options: 'none', 'sum', 'mean'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The minimum computed loss, the corresponding index, and the used mask.
        """
        losses = force_atoms_to_crystal_structure_loss(
            atoms, self.structure_atom_counts
        )
        losses_summed_over_atoms = torch.sum(losses, dim=2)
        min_loss, min_index = torch.min(losses_summed_over_atoms, dim=0)

        if reduction == "none":
            return min_loss, min_index
        elif reduction == "sum":
            return torch.sum(min_loss), min_index
        elif reduction == "mean":
            return torch.mean(min_loss), min_index
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

    def get_structure_and_mask(self, min_index):
        mask = np.concatenate([self.dummy_mask] * len(min_index), axis=0)
        structure_id = np.array(self.structure_ids).reshape(-1, 1)[
            min_index.detach().cpu().numpy()
        ]
        structure = np.squeeze(
            self.structure_atom_counts.detach().cpu().numpy()[structure_id]
        )
        return mask, structure
