import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from IPython.display import Math, display
from omegaconf import DictConfig
from tqdm import tqdm

from .inverse_pl_module import InvModule4PeriodicTable
from .utils import define_atom_list


def adjust_to_integer_ratios(row):
    """
    Adjust the values in the given row to their smallest integer ratios.

    Parameters:
    - row: A numpy array.

    Returns:
    - A numpy array with adjusted values.
    """
    # Find the smallest non-zero value in the row
    min_nonzero = np.min(row[np.nonzero(row)])
    adjusted_row = np.round(row / min_nonzero, 4)
    return adjusted_row


def sort_datasets_by_difference(
    data: np.lib.npyio.NpzFile,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort datasets based on the difference between optimized_tc and init_tc in descending order.

    Args:
        data (np.lib.npyio.NpzFile): Loaded npz file containing the datasets.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Sorted datasets ('optimized_solutions', 'optimized_tc', 'init_solutions', 'init_tc').
    """
    diff = data["optimized_tc"] - data["init_tc"]
    sorted_indices = np.argsort(diff)[::-1]  # Sort in descending order

    return (
        data["optimized_solutions"][sorted_indices],
        data["optimized_tc"][sorted_indices],
        data["init_solutions"][sorted_indices],
        data["init_tc"][sorted_indices],
    )


def sort_datasets_by_optimized_tc(
    data: np.lib.npyio.NpzFile,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort datasets based on the optimized_tc and.

    Args:
        data (np.lib.npyio.NpzFile): Loaded npz file containing the datasets.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Sorted datasets ('optimized_solutions', 'optimized_tc', 'init_solutions', 'init_tc').
    """
    sorted_indices = np.argsort(data["optimized_tc"])[::-1]  # Sort in descending order

    return (
        data["optimized_solutions"][sorted_indices],
        data["optimized_tc"][sorted_indices],
        data["optimized_ef"][sorted_indices],
        data["init_solutions"][sorted_indices],
        data["init_tc"][sorted_indices],
        data["init_ef"][sorted_indices],
    )


def create_atomic_strings(
    solution_adjusted: np.ndarray, atomic_number_to_symbol: dict
) -> List[str]:
    """
    Create atomic strings for the given solutions.

    Args:
        solution_adjusted (np.ndarray): Adjusted solutions.

    Returns:
        List[str]: List of atomic strings.
    """
    atomic_strings = []
    for row in solution_adjusted:
        atomic_str = ""
        for idx, count in enumerate(row):
            if count > 0:
                atomic_str += f"{atomic_number_to_symbol[idx + 1]}_{{{str(np.round(count, 6))[:8]}}} "
        atomic_strings.append(atomic_str)
    return atomic_strings


class Atom_vec2string_converter:
    def __init__(self):
        self.atom_list = define_atom_list()
        self.atomic_number_to_symbol = {
            idx + 1: symbol for idx, symbol in enumerate(self.atom_list)
        }

    def vector2string(self, x: np.ndarray):
        return create_atomic_strings(x, self.atomic_number_to_symbol)


from collections import OrderedDict


def get_elems_str(row):
    elems_set = list()
    for elem_num in row["Initial Optimized Composition"].strip().split(" "):
        elems_set.append(elem_num.split("_")[0])
    elems_set.sort()
    return str(elems_set)


def display_dataframe_with_latex_composition_subscript(df, top_N=5):
    """
    Display a DataFrame with the composition columns in LaTeX format (with subscript) row by row for top N rows.

    Parameters:
    - df: The DataFrame to display.
    - top_N: Number of top rows to display.
    """
    elems_str_set = set()
    display_dict = OrderedDict()
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Prepare for Displaying..."
    ):
        if (type(row["Initial Optimized Composition"]) is not str) or (
            type(row["Rounded Optimized Composition"]) is not str
        ):
            continue

        # Skip if the combination of elements with higher Tc has already been displayed
        elems_str = get_elems_str(row)
        if elems_str in display_dict.keys():
            latex_str, rounded_opt_tc = display_dict[elems_str]
            new_tc = np.round(row["Rounded Optimized Tc"], 2)
            if new_tc < rounded_opt_tc or new_tc == 0.0 or np.isnan(new_tc):
                continue
        if np.round(row["Rounded Optimized Tc"], 2) == 0.0:
            continue

        elems_str_set.add(elems_str)

        initial_opt_composition = (
            "\mathrm{" + row["Initial Optimized Composition"].replace("^", "_") + "}"
        )
        rounded_opt_composition = (
            "\mathrm{" + row["Rounded Optimized Composition"].replace("^", "_") + "}"
        )
        initial_opt_tc = np.round(row["Initial Optimized Tc"], 2)
        rounded_opt_tc = np.round(row["Rounded Optimized Tc"], 2)
        latex_str = f"Optimized: {initial_opt_composition} (Tc: {initial_opt_tc}) \ → \ Rounded: {rounded_opt_composition} (Tc: {rounded_opt_tc})"
        display_dict[elems_str] = latex_str, rounded_opt_tc

    num_displayed = 0
    for _, (latex_str, tc) in display_dict.items():
        display(Math(latex_str))
        num_displayed += 1
        if num_displayed >= top_N:
            break


def display_optimized_solutions(
    cfg: DictConfig, iv_model: InvModule4PeriodicTable, top_N=50
):
    """
    Display the optimized solutions and their corresponding atomic strings.

    Args:
        cfg (DictConfig): Configuration object containing necessary parameters.
        iv_model (InvModule4PeriodicTable): Inverse problem model.
    """
    file_path = os.path.join(
        cfg.output_dirs.inverse_result_dir, "optimized_solutions.npz"
    )
    optimized_solusition_np, optimized_tc, optimized_ef, _, _, _ = (
        sort_datasets_by_optimized_tc(np.load(file_path))
    )
    optimized_solusition = torch.Tensor(optimized_solusition_np)

    # get suitable structure
    _, min_index = iv_model.loss_calculator.calculate_min_loss(
        optimized_solusition.squeeze()
    )
    mask, structure_atom_count = iv_model.loss_calculator.get_structure_and_mask(
        min_index
    )

    # get atomic strings before rounding
    structure_atom_count = torch.Tensor(structure_atom_count)
    integer_optimized_solution = (
        (optimized_solusition * structure_atom_count.view(-1, 1)).detach().cpu().numpy()
    )
    atomic_strings_init = Atom_vec2string_converter().vector2string(
        integer_optimized_solution
    )

    # マスクが0のところはそのまま処理し、maskが1のところは丸める
    if (mask == 1).all():
        non_flag_interger_solution = np.zeros(integer_optimized_solution.shape).astype(
            np.float32
        )
    else:
        non_flag_interger_solution = integer_optimized_solution * (1 - mask)
        num_atoms = np.sum(non_flag_interger_solution, axis=1, keepdims=True)
        non_flag_interger_solution = (
            non_flag_interger_solution / num_atoms
        ) * np.round(num_atoms)

    # マスクが1のところをroundする
    rounded_optimized_solution = np.round(integer_optimized_solution) * mask

    # 非整数部分と整数部分を組み合わせ、丸めた最適解を得る
    rounded_optimized_solution = non_flag_interger_solution + rounded_optimized_solution
    rounded_optimized_tc = iv_model.sg_model(
        torch.tensor(rounded_optimized_solution / structure_atom_count.view(-1, 1))
        .view(-1, 118, 1, 1, 1)
        .clone()
    )
    rounded_optimized_ef = iv_model.elemnet(
        torch.tensor(rounded_optimized_solution / structure_atom_count.view(-1, 1))
        .view(-1, 118)
        .clone()[:, :86]
    )
    atomic_strings_round = Atom_vec2string_converter().vector2string(
        rounded_optimized_solution
    )

    # Create a DataFrame to display the results
    df = pd.DataFrame(
        {
            "Initial Optimized Composition": atomic_strings_init,
            "Rounded Optimized Composition": atomic_strings_round,
            "Rounded Optimized Tc": rounded_optimized_tc.reshape(-1),
            "Initial Optimized Tc": optimized_tc.reshape(-1),
            "Rounded Optimized Ef": rounded_optimized_ef.reshape(-1),
            "Initial Optimized Ef": optimized_ef.reshape(-1),
            "structure_atom_count": structure_atom_count.detach().cpu().numpy(),
        }
    )
    df.to_csv(
        os.path.join(cfg.output_dirs.inverse_result_dir, "optimized_solutions.csv"),
        index=False,
    )
    if cfg.inverse_problem.display_results:
        display_dataframe_with_latex_composition_subscript(df, top_N)

    top_N_Tc_mean = np.mean(optimized_tc[:top_N])

    return top_N_Tc_mean


def display_latex_command(cfg: DictConfig, iv_model: InvModule4PeriodicTable):
    file_path = os.path.join(
        cfg.output_dirs.inverse_result_dir, "optimized_solutions.npz"
    )
    optimized_solusition_np, optimized_tc, optimized_ef, _, _, _ = (
        sort_datasets_by_optimized_tc(np.load(file_path))
    )
    optimized_solusition = torch.Tensor(optimized_solusition_np)

    # get suitable structure
    _, min_index = iv_model.loss_calculator.calculate_min_loss(
        optimized_solusition.squeeze()
    )
    mask, structure_atom_count = iv_model.loss_calculator.get_structure_and_mask(
        min_index
    )

    # get atomic strings before rounding
    structure_atom_count = torch.Tensor(structure_atom_count)
    integer_optimized_solution = (
        (optimized_solusition * structure_atom_count.view(-1, 1)).detach().cpu().numpy()
    )
    atomic_strings_init = Atom_vec2string_converter().vector2string(
        integer_optimized_solution
    )

    ### 一部に非整数を使うときの処理
    # マスクが0のところはそのまま処理し、maskが1のところは丸める
    if (mask == 1).all():
        non_flag_interger_solution = np.zeros(integer_optimized_solution.shape).astype(
            np.float32
        )
    else:
        non_flag_interger_solution = integer_optimized_solution * (1 - mask)
        num_atoms = np.sum(non_flag_interger_solution, axis=1, keepdims=True)
        non_flag_interger_solution = (
            non_flag_interger_solution / num_atoms
        ) * np.round(num_atoms)
    # マスクが1のところをroundする
    rounded_optimized_solution = np.round(integer_optimized_solution) * mask

    # 非整数部分と整数部分を組み合わせ、丸めた最適解を得る
    rounded_optimized_solution = non_flag_interger_solution + rounded_optimized_solution
    rounded_optimized_tc = iv_model.calculate_tc(
        torch.tensor(rounded_optimized_solution / structure_atom_count.view(-1, 1))
        .view(-1, 118, 1, 1, 1)
        .clone()
    )
    rounded_optimized_ef = iv_model.calculate_ef(
        torch.tensor(rounded_optimized_solution / structure_atom_count.view(-1, 1))
        .view(-1, 118, 1, 1, 1)
        .clone()
    )

    atomic_strings_round = Atom_vec2string_converter().vector2string(
        rounded_optimized_solution
    )

    # Create a DataFrame to display the results
    df = pd.DataFrame(
        {
            "Initial Optimized Composition": atomic_strings_init,
            "Rounded Optimized Composition": atomic_strings_round,
            "Rounded Optimized Tc": optimized_tc,
            "Initial Optimized Tc": rounded_optimized_tc,
            "Rounded Optimized Ef": optimized_ef,
            "Initial Optimized Ef": rounded_optimized_ef,
            "structure_atom_count": structure_atom_count.detach().cpu().numpy(),
        }
    )

    for idx, row in df.head(30).iterrows():
        initial_opt_composition = (
            "\mathrm{" + row["Initial Optimized Composition"].replace("^", "_") + "}"
        )
        rounded_opt_composition = (
            "\mathrm{" + row["Rounded Optimized Composition"].replace("^", "_") + "}"
        )
        initial_opt_tc = np.round(row["Initial Optimized Tc"], 2)
        rounded_opt_tc = np.round(row["Rounded Optimized Tc"], 2)
        latex_str = f"Optimized: {initial_opt_composition} (Tc: {initial_opt_tc}) \ → \ Rounded: {rounded_opt_composition} (Tc: {rounded_opt_tc})"
        print(latex_str)
