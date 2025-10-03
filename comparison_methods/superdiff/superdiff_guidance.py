#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


model_retrain = False


import timm
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
from torchvision.models import (
    convnext_tiny,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    wide_resnet50_2,
    wide_resnet101_2,
)

# from surrogate_models.surrogate_utils import Read_AtomMap, num_input_features
# from surrogate_models.t4pt import create_t4pt
# from surrogate_models.vit4pt import create_vit4pt


def replace_bn_with_in(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            setattr(model, name, nn.InstanceNorm2d(num_features, affine=True))
        else:
            replace_bn_with_in(module)


class CNNBaseModel(nn.Module):
    def __init__(self, model_type: str, num_feat: int):
        super(CNNBaseModel, self).__init__()
        # Load a pre-trained ResNet18
        if model_type.lower() == "resnet18":
            self.base_model = resnet18(pretrained=False)
        elif model_type.lower() == "resnet34":
            self.base_model = resnet34(pretrained=False)
        elif model_type.lower() == "resnet50":
            self.base_model = resnet50(pretrained=False)
        elif model_type.lower() == "resnet101":
            self.base_model = resnet101(pretrained=False)
        elif model_type.lower() == "resnet152":
            self.base_model = resnet152(pretrained=False)
        elif model_type.lower() == "w-resnet50-2":
            self.base_model = wide_resnet50_2(pretrained=False)
        elif model_type.lower() == "w-resnet101-2":
            self.base_model = wide_resnet101_2(pretrained=False)
        elif model_type.lower() == "convnext-tiny":
            self.base_model = convnext_tiny(pretrained=False)
        else:
            raise Exception(f"Invalid model_type: {model_type}")

        self.base_model.conv1 = torch.nn.Conv2d(
            num_feat,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        if model_type.lower() == "convnext-tiny":
            self.base_model.classifier[2] = torch.nn.Linear(
                self.base_model.classifier[2].in_features, 1
            )
        else:
            self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 1)

        self.fin_act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fin_act(self.base_model(x))


class PeriodicTableSurrogateModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Define the model architecture.
        """
        super(PeriodicTableSurrogateModel, self).__init__()
        self.num_feat = num_input_features(
            cfg.sg_model.atom_map_type, cfg.sg_model.model_type
        )
        self.base_model = self.define_base_model(cfg.sg_model.model_type)
        self.atom_map = self.define_AtomMap(cfg)

        self.model_type = cfg.sg_model.model_type

    def x_atom_map_process(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type.startswith("T4PT-"):
            x = x.squeeze(-1).squeeze(-1)
        else:
            x = x.unsqueeze(1)
            x = torch.sum(x * self.atom_map, dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_atom_map_process(x)
        return self.base_model(x)

    def define_AtomMap(self, cfg: DictConfig):
        atom_map = torch.from_numpy(Read_AtomMap(cfg)).float()
        if cfg.sg_model.atom_map_type == "LearnableAtomMap":
            required_grad = True
        else:
            required_grad = False

        atom_map = torch.nn.parameter.Parameter(atom_map, requires_grad=required_grad)
        return atom_map

    def define_base_model(self, model_type: str):
        if (
            model_type.lower().startswith("resnet")
            or model_type.lower().startswith("w-resnet")
            or model_type.lower().startswith("convnext")
        ):
            base_model = CNNBaseModel(model_type, self.num_feat)
        elif model_type.lower() == "swin":
            base_model = timm.create_model(
                "swin_base_patch4_window7_224", pretrained=False, num_classes=1
            )
            base_model.patch_embed.proj = torch.nn.Conv2d(
                self.num_feat, 128, kernel_size=(4, 4), stride=(4, 4)
            )

        elif model_type.startswith("ViT4PT-"):
            base_model = create_vit4pt(model_type)

        elif model_type.startswith("T4PT-"):
            base_model = create_t4pt(model_type)

        else:
            raise ValueError(f"Invalid model_type, your input : {model_type}")

        return base_model


# In[4]:


import os

import numpy as np
import torch
from omegaconf import DictConfig

# from surrogate_models.t4pt import define_t4pt_hyperparams
# from surrogate_models.vit4pt import define_vit4pt_hyperparams


def LearnableAtomMap(cfg: DictConfig):
    model_type = cfg.sg_model.model_type
    if model_type.startswith("ViT4PT-"):
        hparams = define_vit4pt_hyperparams(model_type)
        key = "channels"
    elif model_type.startswith("T4PT-"):
        hparams = define_t4pt_hyperparams(model_type)
        key = "num_total_features"
    else:
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
    elif map_type == "LearnableAtomMap" and model_type.startswith("ViT4PT-"):
        num_input_features = None
    elif map_type == "LearnableAtomMap" and model_type.lower().startswith("resnet"):
        num_input_features = 256
    elif map_type == "LearnableAtomMap" and model_type.lower().startswith("w-resnet"):
        num_input_features = 256
    elif map_type == "atom_vector" and model_type.startswith("T4PT-"):
        num_input_features = None
    else:
        raise Exception(f"func num_input_features  Invalid map_type: {map_type}")
    return num_input_features


# In[5]:


def Read_AtomMap(cfg: DictConfig):
    map_type = cfg.sg_model.atom_map_type
    if map_type.endswith("AtomMap-pg.npy") or map_type.endswith("AtomMap-base.npy"):
        atom_map = np.load(os.path.join(cfg.general.processed_data_dir, map_type))[
            np.newaxis, ...
        ]
    else:
        raise Exception(f"Invalid map_type: {map_type}")

    return atom_map


# In[6]:


import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# from surrogate_models.periodic_table import PeriodicTableSurrogateModel


class LitModel(pl.LightningModule):
    """
    Lightning Module for regression task using either pre-trained Vision Transformer or ResNet18.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.opti_method = cfg.sg_model.optimizer
        self.lr = cfg.sg_model.lr
        if cfg.sg_model.pretrain.model_path is None:
            self.sg_model = PeriodicTableSurrogateModel(cfg)
        else:
            print("load the pre-trained model: ", cfg.sg_model.pretrain.model_path)
            self.sg_model = self.load_pretrained_model(cfg)

        self.atom_map = self.define_AtomMap(cfg)

    def forward(self, x):
        x = self.x_atom_map_process(x)
        return self.sg_model(x)

    def load_pretrained_model(self, cfg: DictConfig):
        # load the mode
        sg_model = PeriodicTableSurrogateModel(cfg)
        # load the parameters
        sg_model.load_state_dict(torch.load(cfg.sg_model.pretrain.model_path))

        return sg_model

    def x_atom_map_process(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = torch.sum(x * self.atom_map, dim=1)
        return x

    def define_AtomMap(self, cfg: DictConfig):
        atom_map = torch.from_numpy(Read_AtomMap(cfg)).float()
        if cfg.sg_model.atom_map_type == "LearnableAtomMap":
            required_grad = True
        else:
            required_grad = False

        atom_map = torch.nn.parameter.Parameter(atom_map, requires_grad=required_grad)
        return atom_map

    def training_step(self, batch, batch_idx):
        (x, _), y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat.view(-1), y)  # MAE loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (x, _), y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat.view(-1), y)  # MAE loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx, dataloader_idx):
        # (x, _), y = batch
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat.view(-1), y)  # MAE loss

        if dataloader_idx == 0:
            dataset_type = "test"
        elif dataloader_idx == 1:
            dataset_type = "supercon_test"
        else:
            dataset_type = f"test_{dataloader_idx}"

        self.log(
            f"{dataset_type}_mae_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx):
        # (x, year), y = batch
        x, y = batch
        y_hat = self(x)

        if dataloader_idx == 0:
            dataset_type = "test"
        elif dataloader_idx == 1:
            dataset_type = "supercon_test"
        else:
            dataset_type = f"test_{dataloader_idx}"

        return {
            "dataset_type": dataset_type,
            "inputs": x.cpu(),
            "targets": y.cpu(),
            "predictions": y_hat.squeeze(-1).cpu(),
        }

    def configure_optimizers(self):
        if self.opti_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.use_sheduler = False
            return optimizer

        elif self.opti_method == "Adam-CA":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs * 2)
            self.use_sheduler = True
            return [optimizer], [scheduler]

        elif self.opti_method == "Adam-LWCA":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=10,
                warmup_start_lr=1e-10,
                max_epochs=self.trainer.max_epochs * 2,
            )
            self.use_sheduler = True
            return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        if self.use_sheduler:
            sch = self.lr_schedulers()
            sch.step()


# In[7]:


from omegaconf import DictConfig, OmegaConf


def default_config() -> DictConfig:
    config = {
        "general": {
            "master_dir": "../../",
            "processed_data_dir": "data/processed",
            "deep_speed": None,  # "deepspeed_stage_2", # None or "deepspeed_stage_2"
        },
        "output_dirs": {},
        "experiment_names": {},
        "dataset": {
            "dataset_name": None,
            "divide_method": [0.05, 0.15],
            "element_division": True,
            "divide_infos": "random",
            "train_balancing": True,
        },
        "sg_model": {
            "model_type": "ResNet18",
            "atom_map_type": "AtomMap-base.npy",
            "fin_act": "relu",
            "optimizer": None,  # "Adam-CA",
            "max_epochs": 200,  # None,
            "batch_size": None,  # 1024
            "lr": None,
            "num_accum_batch": 1,
            "model_name": "baseline_surrogate",
            "elemnet_path": "elemnet.pth",
            "pretrain": {
                "dataset_name": None,  # "JARVIS_OQMD_bandgap", "JARVIS_megnet_bandgap", "JARVIS_multitask"
                "task_name": None,  # "bandgap", "e_form", "delta_e", "stability", "multitask"
                "optimizer": None,  # "Adam-CA",
                "max_epochs": 250,
                "batch_size": 256,
                "lr": None,  # 1e-4,
                "num_accum_batch": 1,
                "model_path": None,
                "ce_loss_weight": 1.0,
                "mae_loss_weight": 1.0,
            },
        },
        "inverse_problem": {
            "method": "NA",  # "NeuralLagrangian",
            "method_parameters": {
                "optimizer": "Adam",
                "optimization_steps": 100,
                "iv_batch_size": 100,
                "iv_lr": 0.01,
                # "iv_lr_decay_factor": 0.1,
                # "iv_lr_decay_patience": 10,
                # "grad_clip": 1.0,
                # "eps": 1e-8,
            },
            "proj": {
                "num_leanable_input": 100,
                "use_candidate_selection": True,
                "reduction_schedule": [1, 0.5, 0.25, 0.125, 0.0625],
            },
            "target_id": 0,
        },
    }
    cfg = OmegaConf.create(config)
    return cfg


# In[8]:


import os
from typing import Union

import omegaconf
from omegaconf import DictConfig, ListConfig


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
            "models/surrogate_model/pretrained",
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


# ## 学習済みCNN モデルのロード
#

# In[9]:


cfg = default_config()
cfg_add_infos(cfg, test_mode=False)
cfg.sg_model.pretrain.model_path = os.path.join(
    cfg.output_dirs.sg_model_dir, cfg.sg_model.model_name + ".pt"
)
sg_model = LitModel(cfg)
sg_model.eval()
# sg_model.to(f'cuda:{gpu_device_number}')
sg_model.to("cuda")


# # ElemNetの学習済みモデルをロード

# In[10]:


import torch
from torch import nn


def parse_architecture(architecture_str):
    """
    例: "1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1" をパース
    この文字列から、それぞれのブロックの
    (units, layers, dropoutフラグ, residualフラグ など) を抜き出し、
    nn.Sequential で構築できるようにする
    """
    blocks = architecture_str.strip().split("-")
    arch_info = []
    for block in blocks:
        # block例: "1024x4D" -> units=1024, layers=4, dropoutあり
        # block例: "64x2"   -> units=64,  layers=2, dropoutなし
        # block例: "1"      -> units=1 (出力層)
        if "x" in block:
            # "1024x4D" のように x で区切る
            # さらに "D" "R" といった文字が含まれるかをチェック
            main_part = block.split("x")  # e.g. ["1024", "4D"]
            units_part = main_part[0]  # "1024" の中にRとかが含まれる可能性も?
            # ここでは正規表現や str.isdigit() で厳密に分ける
            import re

            units_match = re.findall(r"\d+", units_part)
            units = int(units_match[0]) if units_match else 32

            # "4D" -> layers=4, dropoutフラグを読み取り
            tail_part = main_part[1]
            layers_match = re.findall(r"\d+", tail_part)
            layers = int(layers_match[0]) if layers_match else 1
            # dropoutやresidualなど
            dropout_flag = "D" in tail_part
            residual_flag = ("R" in tail_part) or ("B" in tail_part)

            arch_info.append((units, layers, dropout_flag, residual_flag))
        else:
            # 例えば "1" だけの場合（最終層）
            # Rなどが入る場合もあるかもしれない
            import re

            units_match = re.findall(r"\d+", block)
            units = int(units_match[0]) if units_match else 1
            dropout_flag = "D" in block
            residual_flag = "R" in block
            # layers=1で固定
            arch_info.append((units, 1, dropout_flag, residual_flag))
    return arch_info


class ElemNet(torch.nn.Module):
    """
    PyTorchでのネットワーク定義
    """

    def __init__(self, input_dim, architecture_str, activation="relu", dropouts=[]):
        super(ElemNet, self).__init__()
        self.arch_info = parse_architecture(architecture_str)
        self.activation = nn.ReLU

        layers = []
        in_features = input_dim
        # dropouts が指定されている場合、ブロックごとに適用していく想定
        dropout_idx = 0

        for i, (units, num_layers, do_flag, res_flag) in enumerate(self.arch_info):
            for layer_i in range(num_layers):
                fc = nn.Linear(in_features, units)
                nn.init.zeros_(
                    fc.bias
                )  # 初期化方法をslim.layers.fully_connected に合わせる
                nn.init.xavier_uniform_(
                    fc.weight, gain=nn.init.calculate_gain("relu")
                )  # 初期化方法をslim.layers.fully_connected に合わせる
                layers.append(fc)

                # 活性化
                if i == len(self.arch_info) - 1 and layer_i == num_layers - 1:
                    # 最後の層は活性化しない
                    print(
                        f"block {i} layer_i:{layer_i} nn.Linear({in_features}, {units})"
                    )

                else:
                    layers.append(self.activation())
                    print(
                        f"block {i} layer_i:{layer_i} nn.Linear({in_features}, {units}), Activation: {self.activation}"
                    )

                in_features = units

            # ドロップアウト (ブロック定義に D があって、かつ dropouts[] がある場合)
            if do_flag and dropout_idx < len(dropouts):
                p = dropouts[dropout_idx]
                p_dropout = float(f"{np.round(1.0 - p, 2):.2f}")
                layers.append(nn.Dropout(p=p_dropout))  # あとで消す
                print(
                    f"block {i}: nn.Dropout({p_dropout}) (p={p})",
                )

            # ブロックごとに dropout_idx を進める(必須かは好み)
            if do_flag:
                dropout_idx += 1

        # 最後のブロックが出力層(units=1)想定
        # ただし arch_infoの最後で既にunits=1が作られている場合、追加しない
        # ここでは、architecture_strの最後に "-1" と書いてあれば回帰出力とみなす
        # そうでなければ明示的に追加
        if self.arch_info[-1][0] != 1:
            layers.append(nn.Linear(in_features, 1))
            print(f"final block {len(self.arch_info)}: nn.Linear({in_features}, 1)")

        print("len(layers):", len(layers))
        for l_i, layer in enumerate(layers):
            print(f"layer {l_i}: {layer}")

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 最後は (batch_size, 1) になる想定
        out = self.model(x)
        # squeeze して (batch_size,) にしてもよい
        return out.view(-1)


# In[11]:


elemnet = ElemNet(
    86,
    "1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1",
    activation="relu",
    dropouts=[0.8, 0.9, 0.7, 0.8],
)
elemnet.load_state_dict(
    torch.load(os.path.join(cfg.output_dirs.sg_model_dir, cfg.sg_model.elemnet_path))
)
elemnet.eval()
elemnet.to("cuda")


# In[ ]:


# # Diffusion model

# In[12]:


# Imports:

import csv
import functools
import itertools

# defaults
import math
from fractions import Fraction

# Typenotes
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas

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

# from functools import partial
# from einops import rearrange, reduce
# from einops.layers.torch import Rearrange


# In[ ]:


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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

    @torch.no_grad()
    def calculate_epsilon(
        self, x_t: torch.Tensor, t: torch.Tensor, model: nn.Module, **kwargs
    ):
        """
        backward for universal guidance
        """
        epsilon = model(x_t, t, **kwargs)
        return epsilon

    @torch.no_grad()
    def calculate_mean(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor,
        add_noise=False,
    ):
        betas_t = self.extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_bar_t = self.extract(
            torch.sqrt(1.0 - self.alphas_bar), t, x_t.shape
        )
        sqrt_recip_alphas_t = self.extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)
        mean = sqrt_recip_alphas_t * (
            x_t - ((betas_t / sqrt_one_minus_alphas_bar_t) * epsilon)
        )
        if add_noise:
            z = torch.randn_like(x_t)
            variance = torch.sqrt(betas_t) * z
            return mean + variance
        else:
            return mean


# In[17]:


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


# In[18]:


DIFFUSION_TIMESTEPS = 1000
diffusion_model = GaussianDiffusion1D(96, DIFFUSION_TIMESTEPS, "cosine")


# In[19]:


# Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 24  # TODO: test different values - change much bigger actually uses CPU (change to like 12)
NO_EPOCHS = 100
PRINT_FREQUENCY = 10
LR = 1e-4
VERBOSE = True
USE_VALIDATION_SET = True


# In[20]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(
    f'NOTE: Using Device: "{device}"',
    "|",
    (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"),
)


# In[ ]:


# In[ ]:


# # 学習と逆拡散過程によるサンプリングのコード

# In[21]:


import os
from typing import Optional

model_retrain = False


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
    model_dir: str = "./models/SuperDiff/",
):
    if fine_tune:
        assert unet is not None
        best_model_path = (
            f"best_unet_param_ELEM-{NUM_TYPE_OF_ELEMENTS}_SC-{TYPE_OF_DATA}_FT.pth"
        )
    else:
        unet = Unet1D(dim=48, dim_mults=(1, 2, 3, 6), channels=1)
        unet.to(device)
        best_model_path = (
            f"best_unet_param_ELEM-{NUM_TYPE_OF_ELEMENTS}_SC-{TYPE_OF_DATA}.pth"
        )

    best_model_path = os.path.join(model_dir, best_model_path)

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

            # torch.save(unet.state_dict(), f"cuprate5_unet_param_{epoch}.pth") # save UNet states - use lowest loss UNet for sampling
            # torch.save(diffusion_model, "diffusion_model.pth") # save diffusion model

            if best_validation_loss > np.mean(mean_epoch_loss_val):
                best_validation_loss = np.mean(mean_epoch_loss_val)
                torch.save(unet.state_dict(), best_model_path)
                # torch.save(diffusion_model, f"best_diffusion_model_ELEM-{NUM_TYPE_OF_ELEMENTS}.pth") # save diffusion model

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


# In[22]:


import itertools
from math import pi

# This code was taken from: https://github.com/jychoi118/ilvr_adm/blob/main/resizer.py
import numpy as np
import torch
from IPython.display import clear_output
from torch import nn

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[23]:


class Resizer(nn.Module):
    def __init__(
        self,
        in_shape,
        scale_factor=None,
        output_shape=None,
        kernel=None,
        antialiasing=True,
    ):
        super(Resizer, self).__init__()

        # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
        scale_factor, output_shape = self.fix_scale_and_size(
            in_shape, output_shape, scale_factor
        )

        # Choose interpolation method, each method has the matching kernel size
        method, kernel_width = {
            "cubic": (cubic, 4.0),
            "lanczos2": (lanczos2, 4.0),
            "lanczos3": (lanczos3, 6.0),
            "box": (box, 1.0),
            "linear": (linear, 2.0),
            None: (cubic, 4.0),  # set default interpolation method as cubic
        }.get(kernel)

        # Antialiasing is only used when downscaling
        antialiasing *= np.any(np.array(scale_factor) < 1)

        # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
        sorted_dims = np.argsort(np.array(scale_factor))
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
        field_of_view_list = []
        weights_list = []
        for dim in self.sorted_dims:
            # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
            # weights that multiply the values there to get its result.
            weights, field_of_view = self.contributions(
                in_shape[dim],
                output_shape[dim],
                scale_factor[dim],
                method,
                kernel_width,
                antialiasing,
            )

            # convert to torch tensor
            weights = torch.tensor(weights.T, dtype=torch.float32)

            # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
            # tmp_im[field_of_view.T], (bsxfun style)
            weights_list.append(
                nn.Parameter(
                    torch.reshape(
                        weights, list(weights.shape) + (len(scale_factor) - 1) * [1]
                    ),
                    requires_grad=False,
                )
            )
            field_of_view_list.append(
                nn.Parameter(
                    torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long),
                    requires_grad=False,
                )
            )

        self.field_of_view = nn.ParameterList(field_of_view_list)
        self.weights = nn.ParameterList(weights_list)

    def forward(self, in_tensor):
        x = in_tensor

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
            # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
            x = torch.transpose(x, dim, 0)

            # This is a bit of a complicated multiplication: x[field_of_view.T] is a tensor of order image_dims+1.
            # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
            # only, this is why it only adds 1 dim to 5the shape). We then multiply, for each pixel, its set of positions with
            # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
            # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
            # same number
            x = torch.sum(x[fov] * w, dim=0)

            # Finally we swap back the axes to the original order
            x = torch.transpose(x, dim, 0)

        return x

    def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
        # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
        # same size as the number of input dimensions)
        if scale_factor is not None:
            # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
            if np.isscalar(scale_factor) and len(input_shape) > 1:
                scale_factor = [scale_factor, scale_factor]

            # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
            scale_factor = list(scale_factor)
            scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor

        # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
        # to all the unspecified dimensions
        if output_shape is not None:
            output_shape = list(input_shape[len(output_shape) :]) + list(
                np.uint(np.array(output_shape))
            )

        # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
        # sub-optimal, because there can be different scales to the same output-shape.
        if scale_factor is None:
            scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

        # Dealing with missing output-shape. calculating according to scale-factor
        if output_shape is None:
            output_shape = np.uint(
                np.ceil(np.array(input_shape) * np.array(scale_factor))
            )

        return scale_factor, output_shape

    def contributions(
        self, in_length, out_length, scale, kernel, kernel_width, antialiasing
    ):
        # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
        # such that each position from the field_of_view will be multiplied with a matching filter from the
        # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
        # around it. This is only done for one dimension of the image.

        # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
        # 1/sf. this means filtering is more 'low-pass filter'.
        fixed_kernel = (
            (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        )
        kernel_width *= 1.0 / scale if antialiasing else 1.0

        # These are the coordinates of the output image
        out_coordinates = np.arange(1, out_length + 1)

        # since both scale-factor and output size can be provided simulatneously, perserving the center of the image requires shifting
        # the output coordinates. the deviation is because out_length doesn't necesary equal in_length*scale.
        # to keep the center we need to subtract half of this deivation so that we get equal margins for boths sides and center is preserved.
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2

        # These are the matching positions of the output-coordinates on the input image coordinates.
        # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
        # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
        # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
        # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
        # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
        # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
        # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
        # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)

        # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
        left_boundary = np.floor(match_coordinates - kernel_width / 2)

        # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
        # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
        expanded_kernel_width = np.ceil(kernel_width) + 2

        # Determine a set of field_of_view for each each output position, these are the pixels in the input image
        # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
        # vertical dim is the pixels it 'sees' (kernel_size + 2)
        field_of_view = np.squeeze(
            np.int16(
                np.expand_dims(left_boundary, axis=1)
                + np.arange(expanded_kernel_width)
                - 1
            )
        )

        # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
        # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
        # 'field_of_view')
        weights = fixed_kernel(
            1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1
        )

        # Normalize weights to sum up to 1. be careful from dividing by 0
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

        # We use this mirror structure as a trick for reflection padding at the boundaries
        mirror = np.uint(
            np.concatenate(
                (np.arange(in_length), np.arange(in_length - 1, -1, step=-1))
            )
        )
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

        # Get rid of  weights and pixel positions that are of zero weight
        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

        # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
        return weights, field_of_view


# These next functions are all interpolation methods. x is the distance from the left pixel center


def cubic(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * ((1 < absx) & (absx <= 2))


def lanczos2(x):
    return (
        (np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps)
        / ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps)
    ) * (abs(x) < 2)


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (
        (np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps)
        / ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps)
    ) * (abs(x) < 3)


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


# In[24]:


def Phi1D(vectors, scale_factor, device):
    """
    Phi_N linear low pass filtering operation function proposed in ILVR paper.

    :param vectors: vectors input to be changed
    :param scale_factor: N, the scale factor

    :return:
    """
    output_vector = torch.empty((0,)).to(device)

    assert vectors.size(2) == diffusion_model.sequence_length

    for i in range(vectors.size(0)):
        operation_vector = vectors[i]
        resized_vector = operation_vector.repeat(96, 1).unsqueeze(0)

        shape = (1, 1, resized_vector.size(1), resized_vector.size(2))
        shape_d = (
            1,
            1,
            int(resized_vector.size(1) / scale_factor),
            int(resized_vector.size(2) / scale_factor),
        )
        down = Resizer(shape, 1 / scale_factor).to(device)
        up = Resizer(shape_d, scale_factor).to(device)
        resizers = (down, up)

        scaled_vector = up(down(resized_vector.unsqueeze(1).to(device))).squeeze()
        scaled_vector = scaled_vector[0].unsqueeze(0)
        output_vector = torch.cat((output_vector, scaled_vector), 0)

    return output_vector


def define_elemnet_mask() -> torch.Tensor:
    elemnet_atom_ids = [
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        88,
        89,
        90,
        91,
        92,
        93,
    ]
    elemnet_mask = torch.zeros(118).view(1, 118)
    elemnet_mask[:, elemnet_atom_ids] = 1.0
    return elemnet_mask


def calculate_tc_ef_loss(vectors, sg_model, elemnet, ef_strength, return_loss=True):
    # ここから Classifier guidance
    ### ElemNetの制限により、112原子までのベクトルを使う
    vectors4eval = vectors.clone()
    vectors4eval[:, :, 112:] = 0
    ### 0以上のclipする
    vectors4eval = torch.clip(vectors4eval, min=0)
    ### 合計1に規格化し、float32に変換
    norm_coef = torch.sum(vectors4eval, dim=2, keepdim=True)
    normed_atom_vectors = (vectors4eval / norm_coef).squeeze().float()  # shape: [B, 96]
    ### Tc予測モデル用に、0で埋めたベクトルを入れて、shape: [B, 118]へと拡張する
    dummy_vector = torch.zeros((normed_atom_vectors.size(0), 22)).to(device)
    normed_atom_vectors_for_tc_pred = torch.cat(
        (normed_atom_vectors, dummy_vector), dim=1
    )
    ### ElemNet用のマスクでshape: [B, 86]へと削減する
    elemnet_mask = define_elemnet_mask().to(device).squeeze().to(torch.bool)
    normed_atom_vectors_for_elemnet = normed_atom_vectors[
        :, elemnet_mask[: normed_atom_vectors.shape[1]]
    ]  # 学習データ自体を整形する方向でいきたい
    ### 各モデルでの予測
    tc_pred = sg_model(normed_atom_vectors_for_tc_pred)
    ef_pred = elemnet(normed_atom_vectors_for_elemnet)
    if return_loss:
        ### Tcは最大化、Efは最小化
        loss = -tc_pred.squeeze() + ef_strength * ef_pred.squeeze()
        loss = loss.mean()

        return loss
    else:
        return tc_pred


# In[56]:


def record_tc_ef_history(vectors, sg_model, elemnet, tc_history, ef_history):
    with torch.no_grad():
        # ここから Classifier guidance
        ### ElemNetの制限により、86原子までのベクトルを使う
        vectors4eval = vectors.clone()
        vectors4eval[:, :, 112] = 0
        ### 0以上のclipする
        vectors4eval = torch.clip(vectors4eval, min=0)
        ### 合計1に規格化し、float32に変換
        norm_coef = torch.sum(vectors4eval, dim=2, keepdim=True)
        normed_atom_vectors = (
            (vectors4eval / norm_coef).squeeze().float()
        )  # shape: [B, 96]
        ### Tc予測モデル用に、0で埋めたベクトルを入れて、shape: [B, 118]へと拡張する
        dummy_vector = torch.zeros((normed_atom_vectors.size(0), 22)).to(device)
        normed_atom_vectors_for_tc_pred = torch.cat(
            (normed_atom_vectors, dummy_vector), dim=1
        )
        ### ElemNet用のマスクでshape: [B, 86]へと削減する
        elemnet_mask = define_elemnet_mask().to(device).squeeze().to(torch.bool)
        normed_atom_vectors_for_elemnet = normed_atom_vectors[
            :, elemnet_mask[: normed_atom_vectors.shape[1]]
        ]  # 学習データ自体を整形する方向でいきたい
        ### 各モデルでの予測
        tc_pred = sg_model(normed_atom_vectors_for_tc_pred)
        ef_pred = elemnet(normed_atom_vectors_for_elemnet)
        # 記録
        tc_history.append(tc_pred.squeeze().detach().cpu().numpy())
        ef_history.append(ef_pred.squeeze().detach().cpu().numpy())


"""
Distrontium ruthenate:

Sr2RuO4
"""
import ast
import json

from tqdm.auto import tqdm

# Diffusion modelによる推論のハイパーパラメータ


def DiffusionInference(
    exp_dict: dict,
    element_table: dict,
    validation_element_table: dict,
    unet: nn.Module,
    diffusion_model: GaussianDiffusion1D,
    device: str,
    use_ilvr: bool,
    use_classifier_guidance: bool,
    guidance_learning_rate: float,
    save_history: bool = False,
    num_step_m: Optional[int] = None,
    Delta_lr: Optional[float] = None,
    ug_reccurent_steps_k: int = 5,
    self_recrusive: bool = True,
    grad_coef_for_test=1,
    result_dir: str = "./",
    ef_strength: float = 4.0,
    test_mode: bool = False,
):
    # 組成式を得る
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
        merge_sc_char(split_scform_to_char(sro_set[0])), element_table
    )
    # nickelate_set_2 = split_sc_to_vector(merge_sc_char(split_scform_to_char(nickelate_set[1])), element_table)
    sro_set_1 = torch.from_numpy(sro_set_1)
    # nickelate_set_2 = torch.from_numpy(nickelate_set_2)
    # nickelate_set = torch.stack((nickelate_set_1, nickelate_set_2))
    sro_set = sro_set_1
    print(sro_set)
    print(cform_from_vector(sro_set, element_table))

    """
    IVLR Modifications (conditioning set)
    """
    # reference_set = []
    # TEMP reference_set
    reference_set = sro_set.unsqueeze(0)
    # PARAMS
    VERBOSE_SAMPLING = True
    available_scale_factors = (2, 4, 8, 16, 32)  # TODO: optimize for 96 max 32
    # scale_factors = (4, 48, 16, 32)
    # scale_factors = (3, 6, 12, 24)
    if test_mode or save_history:
        scale_factors = (2,)
    else:
        scale_factors = (2, 3, 4, 6)
    NUM_SAMPLE_PER_FACTOR = 1024
    NUM_SAMPLE_VECTORS = NUM_SAMPLE_PER_FACTOR * len(scale_factors) * len(reference_set)

    # history
    tc_history, ef_history = [], []

    # sampling
    total_generated_output = torch.empty((0,)).to(
        device
    )  # ensure that on same device as "vectors"
    print(total_generated_output.device.type)  # ensure that on same device as "vectors"
    print(
        f"Sampling {int(len(scale_factors) * NUM_SAMPLE_PER_FACTOR)} samples for {int(len(reference_set))} references."
    )
    for i in tqdm(
        range(int(len(reference_set))), total=len(reference_set), desc="Reference loop"
    ):
        scale_factor_generated_output = torch.empty((0,)).to(device)
        for j in tqdm(
            range(int(len(scale_factors))),
            total=len(scale_factors),
            desc="Scale factor loop",
            leave=False,
        ):
            vectors = torch.randn(
                NUM_SAMPLE_PER_FACTOR, 1, diffusion_model.sequence_length
            ).to(device)
            for k in tqdm(
                reversed(range(diffusion_model.timesteps)),
                total=diffusion_model.timesteps,
                desc="Diffusion steps",
            ):
                t = torch.full((1,), k, dtype=torch.long, device=device)

                """
                Iterative Latent Variable Refinement (IVLR)
                Algorithm Implementation (https://arxiv.org/pdf/2108.02938v2.pdf):
                
                Same DDPM Training (applying conditioning after training)
                Input: Reference Vector $y$: reference_set[i]
                Output: Generated Vector $x$: appended to scale_factor_generated_output
                """
                vectors = vectors.to(torch.float32)
                if not use_classifier_guidance:
                    if k == diffusion_model.timesteps - 1:
                        print(f"No Guidance")
                    # vectors_prime = diffusion_model.backward(vectors.to(torch.float32), t, unet.eval().to(device))
                    epsilon = diffusion_model.calculate_epsilon(
                        vectors, t, unet.eval().to(device)
                    )
                    vectors_prime = diffusion_model.calculate_mean(
                        vectors.detach(), t, epsilon, add_noise=True
                    )

                    if save_history:
                        record_tc_ef_history(
                            vectors_prime, sg_model, elemnet, tc_history, ef_history
                        )
                        print(f"Tc@step{k} before guide", tc_history[-1].mean())

                elif (
                    use_classifier_guidance == "UG" and k == 0
                ):  # t=0での処理がよくわからんので、t=0では処理しない
                    pass
                elif (
                    use_classifier_guidance == "UG" and k > 0
                ):  # t=0での処理がよくわからんので、t=0では処理しない
                    if k == diffusion_model.timesteps - 1:
                        print(
                            f"Universal Guidance. K={ug_reccurent_steps_k}, m={num_step_m} and its learning rate={Delta_lr}"
                        )
                    # Tc, Ef 記録のために、Universal Guidanceをかけるまえの組成式vectorsを記録しておく
                    pre_ug_vectors = vectors

                    # k回の更新の処理
                    for _ in range(ug_reccurent_steps_k):
                        # Diffusion model からepsilonを計算する
                        epsilon = diffusion_model.calculate_epsilon(
                            vectors, t, unet.eval().to(device)
                        )

                        # Universal Guidance
                        ## vectorsを学習可能にする
                        z_t = vectors.detach().float().requires_grad_(True)
                        ## 更新の幅、つまりoptimizer の係数を取得する
                        beta_t = (
                            diffusion_model.extract(
                                diffusion_model.betas, t, vectors.shape
                            )
                        ).float()
                        alpha_t = 1.0 - beta_t
                        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
                        s_t = (
                            sqrt_one_minus_alpha * guidance_learning_rate
                        )  #  guidance_learning_rate はここではw
                        print(k, "s(t):", s_t)

                        ## Eq (3)
                        z_hat_zero = (
                            z_t - torch.sqrt(1.0 - alpha_t) * epsilon
                        ) / torch.sqrt(alpha_t)
                        ## Eq.(6)
                        ### calculate loss and gradients
                        loss = calculate_tc_ef_loss(
                            z_hat_zero, sg_model, elemnet, ef_strength
                        )
                        grad = torch.autograd.grad(loss, z_t)[0]
                        ### calculate epsilon_hat
                        epsilon_hat = epsilon + s_t * grad
                        ## Eq.(7) and (8)
                        ## num_step_m times gradient
                        if num_step_m is not None:
                            ### Eq.(7)
                            Delta = (
                                torch.zeros_like(z_hat_zero)
                                .float()
                                .requires_grad_(True)
                            )
                            optimizer_Delta = torch.optim.SGD([Delta], lr=Delta_lr)
                            for _ in range(num_step_m):
                                loss = calculate_tc_ef_loss(
                                    Delta + z_hat_zero.detach().clone(),
                                    sg_model,
                                    elemnet,
                                    ef_strength,
                                )
                                loss.backward()
                                optimizer_Delta.step()
                            Delta_z_zero = Delta.detach().clone()
                            ### Eq. (8)
                            epsilon_hat = (
                                epsilon_hat
                                - torch.sqrt(alpha_t / (1 - alpha_t)) * Delta_z_zero
                            )

                        ## self-recurrence
                        if self_recrusive:
                            ## z_(t-1) <- S(z_t, epsilon_hat, t)
                            z_t_minus_one = diffusion_model.calculate_mean(
                                z_t.detach(), t, epsilon_hat
                            )
                            ### noise
                            noise = torch.randn_like(z_t).float()
                            ### calculate z_t
                            alpha_t_minus_one = 1.0 - diffusion_model.extract(
                                diffusion_model.betas, t - 1, vectors.shape
                            )
                            z_t = (
                                torch.sqrt(alpha_t / alpha_t_minus_one) * z_t_minus_one
                                + torch.sqrt(1.0 - alpha_t / alpha_t_minus_one) * noise
                            )
                            # z_tをvectorsに代入する
                            vectors = z_t.clone().float()
                        else:
                            vectors = diffusion_model.calculate_mean(
                                z_t.detach(), t, epsilon_hat, add_noise=True
                            ).float()
                    # K回の更新処理の終わり
                    vectors_prime = vectors.clone()

                    # Tc, Efの変化を記録する
                    if save_history:
                        # UGなしで逆拡散したときのTcを記録
                        epsilon = diffusion_model.calculate_epsilon(
                            pre_ug_vectors, t, unet.eval().to(device)
                        )
                        vectors_no_ug = diffusion_model.calculate_mean(
                            pre_ug_vectors.detach(), t, epsilon, add_noise=True
                        )
                        record_tc_ef_history(
                            vectors_no_ug, sg_model, elemnet, tc_history, ef_history
                        )
                        print(f"Tc@step{k} before guide", tc_history[-1].mean())
                        # UGあとのTcを記録する
                        record_tc_ef_history(
                            vectors_prime, sg_model, elemnet, tc_history, ef_history
                        )
                        print(f"Tc@step{k} after guide", tc_history[-1].mean())
                    # Tc, Efの変化記録おわり
                elif use_classifier_guidance == "CG":
                    # Diffusion model からepsilonを計算する
                    epsilon = diffusion_model.calculate_epsilon(
                        vectors, t, unet.eval().to(device)
                    )
                    vectors_prime = diffusion_model.calculate_mean(
                        vectors.detach(), t, epsilon, add_noise=True
                    )
                    ## Tc, Ef 記録のために、Universal Guidanceをかけるまえの組成式vectorsを記録しておく
                    vectors_pre_cg = vectors_prime

                    # Guidance
                    vectors4grad = vectors_prime.float().requires_grad_(True)
                    loss = calculate_tc_ef_loss(
                        vectors4grad, sg_model, elemnet, ef_strength
                    )
                    grad = torch.autograd.grad(loss, vectors4grad)[0]

                    vectors_prime = vectors_prime - guidance_learning_rate * grad
                    if save_history:
                        record_tc_ef_history(
                            vectors_pre_cg, sg_model, elemnet, tc_history, ef_history
                        )
                        print(f"Tc@step{k} before guide", tc_history[-1].mean())
                        # CGあとのTcを記録する
                        record_tc_ef_history(
                            vectors_prime, sg_model, elemnet, tc_history, ef_history
                        )
                        print(f"Tc@step{k} after guide", tc_history[-1].mean())
                else:
                    raise Exception(
                        f"use_classifier_guidance is invalid: {use_classifier_guidance}"
                    )

                # ILVR
                if use_ilvr:
                    # Compute y_{t-1} ~ q(y_{t-1}|y): conditional encoding of reference image
                    conditional_encoding_vector, _ = diffusion_model.forward(
                        reference_set[i].to(torch.float32), t, device
                    )
                    # resize to [B, 1, L]
                    conditional_encoding_vector = torch.stack(
                        [conditional_encoding_vector.squeeze()] * NUM_SAMPLE_PER_FACTOR
                    ).unsqueeze(1)
                    # TODO: Make sure t input is correct (size and value)
                    # TEMP: SIZE analysis: v' ([16, 1, 96]); y ([1, 1, 96]) stack 16 -> [16, 1, 96];

                    # Compute x_{t-1} <- \phi_{N}(y_{t-1}) + x'_{t-1} - \phi_{N}(x'_{t-1})
                    # NOTE: Theoretically, 0 loss should give \phi_{N}(y_{t-1}) = \phi_{N}(x'_{t-1})
                    resized_conditional = Phi1D(
                        conditional_encoding_vector, scale_factors[j], device
                    ).unsqueeze(1)
                    resized_proposal = Phi1D(
                        vectors_prime, scale_factors[j], device
                    ).unsqueeze(1)
                    vectors = resized_conditional + vectors_prime - resized_proposal
                    if save_history:
                        record_tc_ef_history(
                            vectors, sg_model, elemnet, tc_history, ef_history
                        )
                else:
                    vectors = vectors_prime

                if k == 0:
                    print("t = 0")
                    print(
                        f"Scale Factor: Sample {j + 1} out of {len(scale_factors)} complete."
                    )

            # concatenate to scale_factor_generated_output
            scale_factor_generated_output = torch.cat(
                (scale_factor_generated_output, vectors), 0
            )

        print(f"Reference Set: Sample {i + 1} out of {len(reference_set)} complete.")

        # concatenate to total
        total_generated_output = torch.cat(
            (total_generated_output, scale_factor_generated_output), 0
        )

    print(
        f"Sampling Complete. Sampled a total of {int(NUM_SAMPLE_PER_FACTOR * len(scale_factors) * len(reference_set))} vectors."
    )

    if use_ilvr:
        result_string = (
            f"sampled_data_from__{cform_from_vector(sro_set, element_table)}"
        )
    else:
        result_string = "random_sampling"

    if use_classifier_guidance == "UG":
        result_string += f"___K{ug_reccurent_steps_k}-w{guidance_learning_rate}"
        if num_step_m is not None:
            result_string += f"_m{num_step_m}-lr{Delta_lr}"

    npz_path = os.path.join(result_dir, f"{result_string}.npz")

    np.savez(
        npz_path,
        total_generated_output=total_generated_output.detach().cpu().numpy(),
        tc_history=np.array(tc_history),
        ef_history=np.array(ef_history),
    )
    return npz_path


# # 計算実行

# In[57]:

# In[57]:


## Element Table
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


# In[60]:

import glob

# ハイパラの設定
test_mode = False
## Classifier guidance
use_classifier_guidance = "CG"
guidance_learning_rate = 0.001
## どの学習済みモデルを使うか
fine_tune = False


## 元コードでのハイパラ
DIFFUSION_TIMESTEPS = 2 if test_mode else 1000
# Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 24  # TODO: test different values - change much bigger actually uses CPU (change to like 12)
NO_EPOCHS = 100
PRINT_FREQUENCY = 10
SUPERCON_DATA_FILE = "./SuperCon_with_types.dat"
LR = 1e-1
VERBOSE = True
USE_VALIDATION_SET = True
NUM_TYPE_OF_ELEMENTS = 96
diffusion_model = GaussianDiffusion1D(96, DIFFUSION_TIMESTEPS, "cosine")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(
    f'NOTE: Using Device: "{device}"',
    "|",
    (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"),
)

specific_or_unconditional = "specific"  # specific"" or "unconditional"

for json_path in glob.glob("./ref_mat/*.json"):
    for guidance_learning_rate, use_classifier_guidance in zip(
        [0.001, None], ["CG", False]
    ):
        assert specific_or_unconditional in ["specific", "unconditional"]

        sc_type_name = (
            json_path.split("/")[-1]
            .replace("experiment_", "")
            .replace("_dict.json", "")
        )

        if use_classifier_guidance:
            result_dir = f"./results/{sc_type_name}/with_classifier_guidance__lr-{guidance_learning_rate}__{specific_or_unconditional}-model"
        else:
            result_dir = f"./results/{sc_type_name}/without_classifier_guidance__{specific_or_unconditional}-model"

        os.makedirs(result_dir, exist_ok=True)
        # jsonファイルを読み込み、実験対象の材料を取得する
        with open(json_path, "r") as f:
            experiment_dict = json.load(f)
        for exp_dict in experiment_dict:
            # SCの種類によって、モデルの種類を変更する
            ## SC種の特定
            if (
                exp_dict["sc_type"] == "CuO-SC"
                and specific_or_unconditional == "specific"
            ):
                TYPE_OF_DATA = "cuprate"
            elif (
                exp_dict["sc_type"] == "Fe-SC"
                and specific_or_unconditional == "specific"
            ):
                TYPE_OF_DATA = "pnictide"
            elif (
                exp_dict["sc_type"] == "Others"
                and specific_or_unconditional == "specific"
            ):
                TYPE_OF_DATA = "others"
            elif specific_or_unconditional == "unconditional":
                TYPE_OF_DATA = "unconditional"
            else:  # Unconditional
                raise Exception("Unconditional SC is not supported")

            ## SC種の特定モデルのロード
            best_model_path = train_diffusion_model(
                96,  # NUM_TYPE_OF_ELEMENTSは96固定、86でやってもデータは削減されなかったので、そのまま使う
                TYPE_OF_DATA,
                False,  # model_retrain
                -1,  # dummy
                DataLoader([]),  # dummy,
                DataLoader([]),  # dummy,
                device,
                fine_tune=fine_tune,
                model_dir="./models",
            )
            unet = Unet1D(dim=48, dim_mults=(1, 2, 3, 6), channels=1)
            unet.to(device)
            unet.load_state_dict(torch.load(best_model_path))

            DiffusionInference(
                exp_dict=exp_dict,
                element_table=element_table,
                validation_element_table=validation_element_table,
                unet=unet,
                diffusion_model=diffusion_model,
                device=device,
                use_ilvr=True,
                use_classifier_guidance=use_classifier_guidance,
                guidance_learning_rate=guidance_learning_rate,
                result_dir=result_dir,
            )
