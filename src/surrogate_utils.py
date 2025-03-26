import os

import numpy as np
import torch
from omegaconf import DictConfig


def Read_AtomMap(cfg: DictConfig):
    map_type = cfg.sg_model.atom_map_type
    if map_type.endswith("AtomMap-pg.npy") or map_type.endswith("AtomMap-base.npy"):
        atom_map = np.load(os.path.join(cfg.general.processed_data_dir, map_type))[
            np.newaxis, ...
        ]
    elif map_type == "LearnableAtomMap":
        atom_map = LearnableAtomMap(cfg)
    elif map_type == "atom_vector":
        atom_map = np.ones((1))
    else:
        raise Exception(f"Invalid map_type: {map_type}")

    return atom_map


def LearnableAtomMap(cfg: DictConfig):
    model_type = cfg.sg_model.model_type
    hparams = {"in_feature": 256}
    key = "in_feature"

    base_map = np.load(os.path.join(cfg.general.processed_data_dir, "AtomMap-base.npy"))
    num_atoms, _, img_h, img_w = base_map.shape

    learn_map_shape = (num_atoms, hparams[key], img_h, img_w)

    return torch.rand(size=learn_map_shape).numpy()


def num_input_features(map_type: str, model_type: str):
    if map_type.endswith("AtomMap-pg.npy"):
        num_input_features = 29
    elif map_type.endswith("AtomMap-base.npy"):
        num_input_features = 4
    else:
        raise Exception(f"func num_input_features  Invalid map_type: {map_type}")
    return num_input_features
