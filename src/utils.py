import ast
import os
from typing import Union

import numpy as np
import omegaconf
import periodictable
import smact
import torch
from omegaconf import DictConfig, ListConfig


def define_atom_list():
    return [periodictable.elements[i].symbol for i in range(1, 119)]


def cfg_add_infos(cfg: DictConfig, test_mode: bool):
    cfg_add_exp_name(cfg, test_mode)
    cfg_add_dirs(cfg)


def cfg_add_exp_name(cfg: DictConfig, test_mode: bool):
    ### dataset name
    dataset_name = define_dataset_name(
        cfg.dataset.divide_method,
        cfg.dataset.divide_infos,
        cfg.dataset.train_balancing,
        cfg.dataset.element_division,
    )
    cfg.dataset.dataset_name = dataset_name
    cfg.experiment_names.surrogate = define_surrogate_experiment_name(cfg, test_mode)
    cfg.experiment_names.inverse = define_periodic_table_inverse_experiment_name(
        cfg, test_mode
    )
    cfg.experiment_names.pretrain_surrogate = define_pretrain_surrogate_experiment_name(
        cfg, test_mode
    )


def define_pretrain_surrogate_experiment_name(
    cfg: DictConfig, test_mode: bool, trial=None
) -> str:
    # dataset
    experiment_name = (
        "ds-" + str(cfg.sg_model.pretrain.dataset_name).replace(".npz", "") + "-"
    )
    experiment_name += str(cfg.sg_model.pretrain.task_name) + "_"

    # model/type
    experiment_name += str(cfg.sg_model.model_type)
    if cfg.sg_model.fin_act is None:
        experiment_name += "-NoAct"
    elif cfg.sg_model.fin_act == "relu":
        pass
    experiment_name += "_"

    # optimizer, learning rate, batch size, deepspeed
    experiment_name += str(cfg.sg_model.pretrain.optimizer) + str(
        cfg.sg_model.pretrain.lr
    )
    if cfg.sg_model.num_accum_batch > 1:
        effective_batch_size = (
            cfg.sg_model.pretrain.num_accum_batch * cfg.sg_model.pretrain.batch_size
        )
        experiment_name += (
            f"_ebs{effective_batch_size}bs{cfg.sg_model.pretrain.batch_size}"
        )
    else:
        experiment_name += "_bs" + str(cfg.sg_model.pretrain.batch_size)

    if cfg.general.deep_speed == "deepspeed_stage_2":
        experiment_name += f"-DS2_"
    else:
        experiment_name += f"_"

    experiment_name += str(cfg.sg_model.atom_map_type).replace(".npy", "")

    if trial is not None:
        experiment_name += "_TRIAL" + str(trial)

    if (cfg.sg_model.pretrain.ce_loss_weight is not None) and (
        cfg.sg_model.pretrain.mae_loss_weight is not None
    ):
        experiment_name += f"_ceW{cfg.sg_model.pretrain.ce_loss_weight}_maeW{cfg.sg_model.pretrain.mae_loss_weight}"

    if test_mode:
        experiment_name += "__TEST"

    return experiment_name


def cfg_add_dirs(cfg: DictConfig):
    """add output directory for a forward model, Inverse problem results on config"""

    # output file saved dirctory
    sg_model_dir = os.path.join(cfg.general.master_dir, "models", "surrogate_model")
    os.makedirs(sg_model_dir, exist_ok=True)

    # pretrained model directory
    if cfg.sg_model.pretrain.model_path is not None:
        cfg.sg_model.pretrain.model_path = os.path.join(
            cfg.general.master_dir,
            "models",
            cfg.sg_model.pretrain.model_path,
        )

    # add new keys on config file
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.general.processed_data_dir = os.path.join(
        cfg.general.master_dir, cfg.general.processed_data_dir
    )
    cfg.output_dirs.surrogate_result_dir = cfg.experiment_names.surrogate
    cfg.output_dirs.inverse_result_dir = cfg.experiment_names.inverse
    cfg.output_dirs.sg_model_dir = sg_model_dir
    cfg.general.devices = torch.cuda.device_count()
    omegaconf.OmegaConf.set_struct(cfg, True)


def define_surrogate_experiment_name(
    cfg: DictConfig, test_mode: bool, trial=None
) -> str:
    # dataset
    experiment_name = "ds-" + str(cfg.dataset.dataset_name).replace(".npz", "") + "_"

    # model/type
    experiment_name += str(cfg.sg_model.model_type)
    if cfg.sg_model.fin_act is None:
        experiment_name += "-NoAct"
    elif cfg.sg_model.fin_act == "relu":
        pass
    experiment_name += "_"

    # optimizer, learning rate, batch size
    experiment_name += str(cfg.sg_model.optimizer) + str(cfg.sg_model.lr)
    if cfg.sg_model.num_accum_batch > 1:
        effective_batch_size = cfg.sg_model.num_accum_batch * cfg.sg_model.batch_size
        experiment_name += f"_ebs{effective_batch_size}bs{cfg.sg_model.batch_size}_"
    else:
        experiment_name += "_bs" + str(cfg.sg_model.batch_size) + "_"

    experiment_name += str(cfg.sg_model.atom_map_type).replace(".npy", "")

    if cfg.sg_model.pretrain.model_path is not None:
        p_model_name = (
            cfg.sg_model.pretrain.model_path.split("/models/")[-1]
            .replace("/", "--")
            .replace(".pt", "")
        )
        experiment_name += f"_pretrained--{p_model_name}"

    if trial is not None:
        experiment_name += "_TRIAL" + str(trial)

    if test_mode:
        experiment_name += "__TEST"

    return experiment_name


def define_periodic_table_inverse_experiment_name(cfg, test_mode: bool) -> str:
    """Return the name of the experiment given the config and test mode"""

    ### Surrogate model settings
    experiment_name = "ds-" + str(cfg.dataset.dataset_name).replace(".npz", "") + "_"
    experiment_name += str(cfg.sg_model.model_type) + "_"
    experiment_name += str(cfg.sg_model.optimizer) + str(cfg.sg_model.lr) + "_"
    experiment_name += str(cfg.sg_model.atom_map_type)

    ### Inverse problem settings
    experiment_name += "___"
    experiment_name += cfg.inverse_problem.method + "_"

    experiment_name += (
        str(cfg.inverse_problem.method_parameters.optimizer)
        + str(cfg.inverse_problem.method_parameters.iv_lr)
        + "-"
    )
    experiment_name += str(cfg.inverse_problem.method_parameters.optimization_steps)

    if test_mode:
        experiment_name += "__TEST"
    return experiment_name


def define_dataset_name(
    divide_method: str,
    divide_infos: list,
    train_balancing: bool,
    element_division: bool,
) -> str:
    """Return the name of the processed data file given the divide method, divide infos and train balancing"""
    file_name = f"{divide_method}-{divide_infos[0]}-{divide_infos[1]}"

    if not element_division:
        file_name += "-NoElementDivision"

    if train_balancing:
        file_name += "-balancing"
    return file_name + ".npz"


def create_atomic_vectors_from_formula_dict(
    formula_dict: dict, val_of_atom: float
) -> np.ndarray:
    """
    Explain: Create atomic vectors from formula dictionary
    Args:
        formula_dict: dict, formula dictionary
    Returns:
        atomic_vector: np.ndarray, atomic vector
    Example:
        formula_dict = {
            "Y": 1.0,
            "Ba": 1.4,
            "Sr": 0.6,
            "Cu": 3.0,
            "O": 6.0,
            "Se": 0.51
        }
        atomic_vector = create_atomic_vectors_from_formula_dict(formula_dict)
    """
    if type(formula_dict) is str:
        formula_dict = ast.literal_eval(formula_dict)
    elif type(formula_dict) is not dict:
        raise ValueError(f"formula_dict must be dict, but got {type(formula_dict)}")

    all_atom_list = define_atom_list()
    atomic_vector = np.array(np.zeros(len(all_atom_list)), dtype=np.float32)
    sum_value = val_of_atom
    for elem, value in formula_dict.items():
        atomic_vector[all_atom_list.index(elem)] = value
        sum_value += value
    atomic_vector = atomic_vector / sum_value  # normalization
    assert (np.sum(atomic_vector) + val_of_atom / sum_value - 1.0) < 1e-5, (
        f"sum of atomic vector is not 1.0, sum={np.sum(atomic_vector)}, val_of_atom={val_of_atom / sum_value}"
    )
    return [float(val) for val in atomic_vector]


def create_oxidation_elem_dict() -> dict:
    """
    Describe: Create oxidation element dictionary
    Args:
        None
    Returns:
        oxidation_dict: dict, oxidation element dictionary
    """
    # create oxidation_dict
    oxidation_dict = {}
    for e_i, elem in enumerate(define_atom_list()[:86]):
        try:
            oxidation_states = smact.Element(elem).oxidation_states
        except NameError:
            print(f"creating oxidation_dict: No.{e_i + 1} Element {elem} is not found.")
            continue

        for ox in oxidation_states:
            if ox not in oxidation_dict.keys():
                oxidation_dict[ox] = []
            oxidation_dict[ox].append(elem)
    return oxidation_dict


def elem_list_to_mask(elem_list) -> np.ndarray:
    """
    Convert element list to mask
    Args:
        elem_list: list, element list
    Returns:
        atomic_vector: np.ndarray, atomic vector
    """
    atomic_vector = np.array(np.zeros(len(define_atom_list())), dtype=np.float32)
    all_atom_list = define_atom_list()
    for elem in elem_list:
        atomic_vector[all_atom_list.index(elem)] = 1.0
    return atomic_vector


def split_scform_to_char(scform_str: str) -> list[str]:
    """
    Split chemical formula string into list of characters.

    :param sc_str: Chemical formula as a string
    """
    split_sc_char = []

    for character in scform_str:
        split_sc_char.append(character)

    return split_sc_char


def split_sc_to_vector(split_sc: list[str], chem_tableDS: list[str]) -> np.ndarray:
    """
    Converts list of chemical elements and quantities to vector in R^(1x96), with index matching elements
    in periodic table from 1 to 96

    :param split_sc: list of chemical elements and associated quantities
    :param chem_tableDS: periodic table of first 96 elements
    """
    sc_vector = np.zeros((1, 96))  # replace size if not R^(8x96)
    for i in range(len(split_sc)):
        if split_sc[i].isalpha() == True:
            for j in range(len(chem_tableDS)):
                if split_sc[i] == chem_tableDS[j]:
                    if i + 1 < len(split_sc):
                        sc_vector[0][j] = float(
                            split_sc[i + 1]
                        )  # split_sc[i+1] contains associated quantity
                    else:
                        sc_vector[0][j] = 1.0  # verify if none at the end = 1.0

    sc_vector = sc_vector.squeeze()  # remove outer parathesis, decrease dimension

    return sc_vector


def merge_sc_char(split_sc_char: list[str]) -> list[str]:
    """
    Merge characters into list of useful element symbols and quantity numbers.

    :param split_sc_char: Chemical formula split by character
    """
    split_sc = []
    temp_numstr = ""
    temp_alphastr = ""

    for i in range(len(split_sc_char)):
        if split_sc_char[i].isalpha() == True:
            if temp_numstr != "":
                split_sc.append(temp_numstr)

            temp_alphastr += split_sc_char[i]
            temp_numstr = ""
        else:
            if temp_alphastr != "":
                split_sc.append(temp_alphastr)
            temp_alphastr = ""
            temp_numstr += split_sc_char[i]

    if temp_numstr != "":
        split_sc.append(temp_numstr)

    if temp_alphastr != "":
        split_sc.append(temp_alphastr)

    return split_sc


def cform_from_vector(form_vector: torch.Tensor, chem_table: list[str]) -> str:
    """
    Helper Function to Turn R^{1x96} Vector back into Chemical Formula

    :param form_vector: Vector encoding of chemical formula
    :param chem_table: Periodic table of elements 1 to 96

    :return: String chemical formula
    """
    cform = ""
    for i in range(0, len(chem_table)):
        if form_vector[i] != 0.0:
            cform += f"{chem_table[i]}{form_vector[i]}"
    return cform


# 組成式を得る
def define_experiment_chemical_formula(exp_dict):
    sroset_str = ""
    for elems, comp in zip(
        ast.literal_eval(exp_dict["elems"]), ast.literal_eval(exp_dict["comp"])
    ):
        if comp.is_integer():
            sroset_str += f"{elems}{round(comp)}"
        else:
            sroset_str += f"{elems}{comp}"
    sro_set = [sroset_str]

    sro_set_1 = split_sc_to_vector(
        merge_sc_char(split_scform_to_char(sro_set[0])), define_atom_list()[:96]
    )
    # nickelate_set_2 = split_sc_to_vector(merge_sc_char(split_scform_to_char(nickelate_set[1])), element_table)
    sro_set_1 = torch.from_numpy(sro_set_1)
    # nickelate_set_2 = torch.from_numpy(nickelate_set_2)
    # nickelate_set = torch.stack((nickelate_set_1, nickelate_set_2))
    sro_set = sro_set_1
    return cform_from_vector(sro_set, define_atom_list()[:96])
