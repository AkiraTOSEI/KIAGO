from typing import Dict, List, Union

import numpy as np
import pandas as pd

from src.utils import define_atom_list


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Mean Absolute Error.
    """
    assert len(y_true) == len(y_pred)
    return np.mean(np.abs(y_true.reshape(-1) - y_pred.reshape(-1)))


def FeSCs_bool(d: Dict[str, np.ndarray], phase: str = "test") -> np.ndarray:
    """
    Return boolean array indicating iron-based superconductors.

    Args:
        d (Dict[str, np.ndarray]): Dictionary containing data.
        phase (str, optional): Phase ('test' by default).

    Returns:
        np.ndarray: Boolean array.
    """
    x = d[f"{phase}_x"].squeeze()
    y_gt = d[f"{phase}_y_gt"].squeeze()
    y_hat = d[f"{phase}_y_hat"].squeeze()
    ib_sc_bool = np.zeros(len(y_gt), dtype=bool)
    atom_list = define_atom_list()
    fe_inx = atom_list.index("Fe")
    for atom in ["As", "S", "Se", "P"]:
        as_inx = atom_list.index(atom)
        ib_sc_bool += (x[:, fe_inx] > 0.0) & (x[:, as_inx] > 0.0) & (y_gt > 0.0)
    return ib_sc_bool


def CuSCs_bool(d: Dict[str, np.ndarray], phase: str = "test") -> np.ndarray:
    """
    Return boolean array indicating copper-based superconductors.

    Args:
        d (Dict[str, np.ndarray]): Dictionary containing data.
        phase (str, optional): Phase ('test' by default).

    Returns:
        np.ndarray: Boolean array.
    """
    x = d[f"{phase}_x"].squeeze()
    y_gt = d[f"{phase}_y_gt"].squeeze()
    y_hat = d[f"{phase}_y_hat"].squeeze()
    atom_list = define_atom_list()
    Cu_inx = atom_list.index("Cu")
    O_inx = atom_list.index("O")
    co_bool = (x[:, Cu_inx] > 0.0) & (x[:, O_inx] > 0.0) & (y_gt > 0.0)

    # other atom exists
    _x = np.sum(x, axis=1)
    _x = _x - x[:, Cu_inx] - x[:, O_inx]
    other_bool = _x > 0.0
    cb_sc_bool = co_bool & other_bool
    return cb_sc_bool


def HySCs_bool(d: Dict[str, np.ndarray], phase: str = "test") -> np.ndarray:
    """
    Return boolean array indicating hydride superconductors.
    The definition of hydrogen superconductivity was taken from the following URL.
    https://www.jstage.jst.go.jp/article/jshpreview/28/4/28_268/_pdf/-char/ja

    Args:
        d (Dict[str, np.ndarray]): Dictionary containing data.
        phase (str, optional): Phase ('test' by default).

    Returns:
        np.ndarray: Boolean array.
    """

    x = d[f"{phase}_x"].squeeze()
    y_gt = d[f"{phase}_y_gt"].squeeze()
    y_hat = d[f"{phase}_y_hat"].squeeze()
    atom_list = define_atom_list()
    H_inx = atom_list.index("H")
    sc_bool = np.zeros(len(y_gt), dtype=bool)
    for atom in [
        "Li",
        "Ca",
        "Y",
        "La",
        "Pd",
        "Pt",
        "Au",
        "Fe",
        "Al",
        "Si",
        "Sn",
        "S",
        "P",
        "Th",
    ]:
        as_inx = atom_list.index(atom)
        sc_bool += (x[:, H_inx] > 0.0) & (x[:, as_inx] > 0.0) & (y_gt > 0.0)
    # 2つの原子のみで構成される化合物
    two_bool = np.sum(x > 0, axis=1) == 2
    sc_bool = sc_bool & two_bool
    return sc_bool


def MAE_PROB_SCORE_BY_TYPE(
    d: Dict[str, np.ndarray], phase: str = "test"
) -> pd.DataFrame:
    """
    Calculate MAE and probability score by superconductor type.

    Args:
        d (Dict[str, np.ndarray]): Dictionary containing data.
        phase (str, optional): Phase ('test' by default).

    Returns:
        pd.DataFrame: DataFrame containing MAE and probability scores.
    """
    results = []
    y_gt = d[f"{phase}_y_gt"].squeeze()
    for sc_type in ["Fe", "Cu", "H", "conventinal", "all"]:
        if sc_type == "Fe":
            sc_bool = FeSCs_bool(d, phase=phase)
            sc_desc = "Fe-based superconductors"
        elif sc_type == "Cu":
            sc_bool = CuSCs_bool(d, phase=phase)
            sc_desc = "Cu-based superconductors"
        elif sc_type == "H":
            sc_bool = HySCs_bool(d, phase=phase)
            sc_desc = "Hydride superconductors"
        elif sc_type == "conventinal":
            sc_bool = (
                (y_gt > 0.0)
                & (~FeSCs_bool(d, phase=phase))
                & (~CuSCs_bool(d, phase=phase))
                & (~HySCs_bool(d, phase=phase))
            )
            sc_desc = "conventional superconductors"
        elif sc_type == "all":
            sc_bool = y_gt > 0.0
            sc_desc = "all superconductors"
        else:
            raise ValueError("sc_type is invalid.")

        mae = calculate_mae(d[f"{phase}_y_gt"][sc_bool], d[f"{phase}_y_hat"][sc_bool])
        results.append([sc_desc.replace(" superconductors", ",MAE"), mae])

        # Probability of predicting Tc>0 for a superconductor
        sc_pred_ratio = np.sum(d[f"{phase}_y_hat"][sc_bool] > 0) / np.sum(sc_bool)
        results.append([sc_desc.replace(" superconductors", ",Prob"), sc_pred_ratio])
    return pd.DataFrame(results, columns=["sc_type", "score"])
