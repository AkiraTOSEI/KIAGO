#!/usr/bin/env python
# coding: utf-8

# # SuperDiffで生成したHSCデータを評価する

# In[1]:


import os

import numpy as np
import pandas as pd

# In[ ]:
# # 評価コード
# ## ElemNetのロード
# In[2]:
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


# In[3]:


elemnet = ElemNet(
    86,
    "1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1",
    activation="relu",
    dropouts=[0.8, 0.9, 0.7, 0.8],
)
elemnet.load_state_dict(torch.load("../../models/surrogate_model/elemnet.pth"))
elemnet.eval()
elemnet.to("cuda")


# In[ ]:


# ## CNNの予測器をロードする

# In[4]:


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


# In[5]:


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


# In[6]:


def Read_AtomMap(cfg: DictConfig):
    map_type = cfg.sg_model.atom_map_type
    if map_type.endswith("AtomMap-pg.npy") or map_type.endswith("AtomMap-base.npy"):
        atom_map = np.load(os.path.join(cfg.general.processed_data_dir, map_type))[
            np.newaxis, ...
        ]
    else:
        raise Exception(f"Invalid map_type: {map_type}")

    return atom_map


# In[7]:


import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
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


# In[8]:


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


# In[9]:


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


# ### 学習済みCNN モデルのロード
#

# In[10]:


cfg = default_config()
cfg_add_infos(cfg, test_mode=False)
cfg.sg_model.pretrain.model_path = os.path.join(
    cfg.output_dirs.sg_model_dir, cfg.sg_model.model_name + ".pt"
)
sg_model = PeriodicTableSurrogateModel(cfg)
sg_model.load_state_dict(torch.load(cfg.sg_model.pretrain.model_path))
sg_model.eval()
# sg_model.to(f'cuda:{gpu_device_number}')
sg_model.to("cuda")


# In[11]:


def define_atom_list():
    atom_list = [
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
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]
    return atom_list


atom_list = define_atom_list()


# In[12]:


import re


def parse_formula(formula: str) -> dict:
    """
    文字列の化学式から要素とその値を辞書にして返す。
    """
    pattern = r"([A-Z][a-z]*)(\d*\.\d+|\d+)?"
    # 例: "H0.16Ta0.92" から
    #     [("H", "0.16"), ("Ta", "0.92"), ("", "")] のように取得されるため
    #     要素と数値が空文字の場合を排除しながら扱う

    composition = {}

    all_atom_list = define_atom_list()

    # 正規表現で要素記号と数値部分を抜き出す
    matches = re.findall(pattern, formula)

    for element, num_str in matches:
        # element や num_str が空の場合はスキップ
        if not element:
            continue

        if num_str == "":
            # 数値が省略されていたら 1.0 とみなす
            value = 1.0
        else:
            value = float(num_str)

        assert element in all_atom_list, f"Invalid element: {element}"

        composition[element] = value

    # ここでキーをソートした辞書を作り直す
    sorted_dict = {k: composition[k] for k in sorted(composition.keys())}
    return sorted_dict


# In[13]:


from itertools import product


# oxidation_statesのN個の組み合わせを求める
def possible_sum_oxidaion_states_of_N_of_Element(N, elem):
    oxidation_states = smact.Element(elem).oxidation_states
    if len(oxidation_states) == 0:
        return []
    return list(
        set(np.sum(np.array(list(product(oxidation_states, repeat=N))), axis=1))
    )


# In[14]:


def possible_sum_oxidaion_states_of_N_of_Hydrogen(N):
    # 各変数  a_i ​ が -1または +1 の値を取り、 N 個あるとすると、1 の数を x とすると全体の和 S は x - (N - x) = 2x - N となる。
    # この S は -N から N までの2ずつ増加する値を取る。
    return list(range(-N, N + 1, 2))


# In[15]:


def get_list_data_from_formula_string(formula):
    elements, comps = [], []
    if formula == "" or type(formula) != str:
        return elements, comps
    for elem_comp in formula.strip().split(" "):
        elem, comp = elem_comp.split("_")
        comp = list(ast.literal_eval(comp))[0]
        elements.append(elem)
        comps.append(round(comp))
    return elements, comps


class check_HSC_neutrality:
    def __init__(self, max_num_atoms, max_num_elements, hydrogen_ratio):
        self.max_num_atoms = max_num_atoms
        self.max_num_elements = max_num_elements
        self.hydrogen_ratio = hydrogen_ratio

    def check_electro_neutrality(self, elements, comps):
        if len(elements) == 0:
            return False, None
        if len(elements) > self.max_num_elements:
            return False, None
        if np.sum(comps) > self.max_num_atoms:
            return False, None
        if "H" not in elements:
            return False, None
        if comps[elements.index("H")] < np.sum(comps) * self.hydrogen_ratio:
            return False, None

        assert len(elements) == len(comps)
        oxidation_combinations = []
        for elem, comp in zip(elements, comps):
            if elem == "H":
                ox_combs = possible_sum_oxidaion_states_of_N_of_Hydrogen(comp)
            else:
                ox_combs = possible_sum_oxidaion_states_of_N_of_Element(comp, elem)

            if len(ox_combs) == 0:
                # Arなど酸化数を含まない元素を含むときはFalseを返す
                return False, None

            oxidation_combinations.append(ox_combs)
        ox_sum_array = np.array(list(product(*oxidation_combinations)))
        ox_sums = np.sum(ox_sum_array, axis=1)
        if 0 in ox_sums:
            ox_states = ox_sum_array[ox_sums == 0][0]
            sample = str(
                {
                    f"{elem} x {comp}": ox
                    for elem, comp, ox in zip(elements, comps, ox_states)
                }
            )
            return True, sample
        else:
            return False, None

    def check_neutrality_from_formula_string(self, formula):
        elements, comps = get_list_data_from_formula_string(formula)
        return self.check_electro_neutrality(elements, comps)


# In[16]:


class check_hydrogen_ratio:
    def __init__(self, hydrogen_thres):
        self.hydrogen_thres = hydrogen_thres

    def check(self, formula):
        elements, comps = get_list_data_from_formula_string(formula)
        if "H" in elements:
            return self.hydrogen_thres <= comps[elements.index("H")] / sum(comps) < 1.0
        else:
            return False


# In[17]:


class num_atoms_check:
    def __init__(self, max_num_atoms):
        self.max_num_atoms = max_num_atoms

    def check(self, formula):
        _, comps = get_list_data_from_formula_string(formula)
        return sum(comps) <= self.max_num_atoms


def create_atomic_vector_from_formula_string(formula):
    elements, comps = get_list_data_from_formula_string(formula)
    atom_vec = np.zeros((118,)).astype(np.float32)
    for elem, comp in zip(elements, comps):
        atom_vec[define_atom_list().index(elem)] = comp
    return atom_vec


# In[18]:


def convert_numpy_to_csv_data(csv_path):
    """
    SuperDiffの出力結果をnpzから読み込んで、HSC評価の形式にする
    """
    npz_path = csv_path.replace(".csv", ".npz")
    total_generated_output = torch.tensor(
        np.load(npz_path, allow_pickle=True)["total_generated_output"]
    ).squeeze()
    total_generated_output = torch.round(
        total_generated_output
    )  # 電気的中性が計算できないので、roundする
    all_atom_vector = total_generated_output.squeeze()
    # normed_all_atom_vector = all_atom_vector/ all_atom_vector.sum(dim=1, keepdim=True)
    # all_atom_vector_for_tc = torch.concat([normed_all_atom_vector, torch.zeros((normed_all_atom_vector.shape[0], 118-normed_all_atom_vector.shape[1]),device=normed_all_atom_vector.device)], dim=1).float()
    # all_atom_vector_for_ef = normed_all_atom_vector[:,:86].float()
    # with torch.no_grad():
    #    all_tc = (sg_model(all_atom_vector_for_tc.view(-1,118,1,1,1).to('cuda'))).detach().cpu().numpy().flatten()
    #    all_ef = (elemnet(all_atom_vector_for_ef.to('cuda'))).detach().cpu().numpy().flatten()
    row_idx, col_idx = np.where(all_atom_vector.detach().cpu().numpy() > 0)
    materials = []
    for sample_i in range(len(all_atom_vector)):
        elems = np.array(element_table)[col_idx[row_idx == sample_i]]
        comps = (
            (all_atom_vector[sample_i, col_idx[row_idx == sample_i]])
            .detach()
            .cpu()
            .numpy()
        )
        mate_str = ""
        for elem, comp in zip(elems, comps):
            mate_str += f"{elem}_" + "{" + f"{comp}" + "} "
        mate_str = mate_str.strip()
        if mate_str == "":
            mate_str = np.nan
        materials.append(mate_str)
    return pd.DataFrame({"neutral_check_formula": materials})


# In[19]:


import ast

import smact
from joblib import Parallel, delayed


def smact_etc_analysis_for_HSC(
    csv_path,
    experiment_type,
    method,
    Ef_criterion,
    elemnet,
    sg_model,
    params_dict,
    hydrogen_thres,
    max_num_atoms,
    max_num_elements,
):
    if method == "proposed":
        original_df = pd.read_csv(csv_path)
        original_df["neutral_check_formula"] = original_df[
            "Rounded Optimized Composition"
        ]
        original_atom_dict = parse_formula(
            csv_path.split("/")[-2]
            .split("___")[0]
            .replace("__results_ef_coef", "")
            .split("-")[0]
        )
    elif method == "superdiff":
        original_df = convert_numpy_to_csv_data(csv_path)
        original_atom_dict = parse_formula(
            csv_path.split("/")[-1].split("__")[1].replace(".csv", "")
        )
        lr = csv_path.split("/")[-2].split("__")[1]
        if lr.startswith("lr"):
            method = f"superdiff-{lr}"
        else:
            method = "superdiff"
    # 一意のもののみを取得する
    unique_df = original_df.drop_duplicates(subset=["neutral_check_formula"]).copy()
    # 電気的中性のチェっっく
    check_neutrality_from_formula_string = check_HSC_neutrality(
        max_num_elements=max_num_elements,
        max_num_atoms=max_num_atoms,
        hydrogen_ratio=hydrogen_thres,
    ).check_neutrality_from_formula_string
    results = Parallel(n_jobs=4)(
        delayed(check_neutrality_from_formula_string)(formula)
        for formula in unique_df["neutral_check_formula"]
    )
    ## 結果を展開して DataFrame に代入
    unique_df["neutrality_bools"], unique_df["neutral_sample"] = zip(*results)
    # 化学式ベクトルから、TcとEfを予測する
    atom_vectors = np.stack(
        unique_df["neutral_check_formula"].apply(
            create_atomic_vector_from_formula_string
        )
    )
    atom_vectors = atom_vectors / np.sum(atom_vectors, axis=1, keepdims=True)
    atom_vectors = torch.tensor(atom_vectors, dtype=torch.float32).to("cuda")
    unique_df["pred_tc"] = (
        sg_model(atom_vectors.view(len(atom_vectors), 118, 1, 1, 1))
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
    unique_df["pred_ef"] = (
        elemnet(atom_vectors[:, :86]).detach().cpu().numpy().flatten()
    )
    # 水素の含有量と原子数の制約を満たしているかをチェック
    hydro_checker = check_hydrogen_ratio(hydrogen_thres).check
    unique_df["hydro_bools"] = unique_df["neutral_check_formula"].apply(hydro_checker)
    num_atoms_checker = num_atoms_check(max_num_atoms).check
    unique_df["num_atoms_bools"] = unique_df["neutral_check_formula"].apply(
        num_atoms_checker
    )
    unique_df["num_elements_bools"] = (
        unique_df["neutral_check_formula"].apply(
            lambda x: len(get_list_data_from_formula_string(x)[0])
        )
        < max_num_elements
    )
    unique_df["valid_sample_bools"] = (
        unique_df["hydro_bools"]
        * unique_df["num_atoms_bools"]
        * unique_df["num_elements_bools"]
    )
    unique_df["Ef_bools"] = unique_df.pred_ef < Ef_criterion
    # 結果をまとめる
    result_df_dict = {
        "formula": unique_df.neutral_check_formula,
        "pred_Tc": unique_df.pred_tc,
        "pred_ef": unique_df.pred_ef,
        "Ef_bools": unique_df.Ef_bools,
        "neutral_bools": unique_df.neutrality_bools.values,
        "hydro_bools": unique_df.hydro_bools,
        "num_atoms_bools": unique_df.num_atoms_bools,
        "valid_sample_bools": unique_df.valid_sample_bools,
    }
    result_df = pd.DataFrame(result_df_dict)
    ## result_df_dict.keys()のなかで、boolsが含まれるものを取得し、すべてのboolsがTrueのサンプルのみを抽出
    bool_keys = [key for key in result_df_dict.keys() if key.endswith("bools")]
    result_df["all_True_bools"] = result_df[bool_keys].all(axis=1)

    # ベース材料情報を得る
    sorted_original_atom_dict = {
        k: original_atom_dict[k] for k in sorted(original_atom_dict.keys())
    }

    # 統計量をdictにまとめる
    summary_dict = {
        "base_atom_dict": str(sorted_original_atom_dict),
        "num_total_samples": len(original_df),
        "unique_rate": len(unique_df) / len(original_df),
        "valid_rate": unique_df["valid_sample_bools"].sum() / len(unique_df),
        "neutral_rate (in valid)": unique_df["neutrality_bools"][
            unique_df["valid_sample_bools"]
        ].sum()
        / unique_df["valid_sample_bools"].sum(),
    }

    # すべてのboolsの達成率を計算
    for b_key in bool_keys + ["all_True_bools"]:
        summary_dict[f"{b_key}_rate"] = result_df[b_key].sum() / len(result_df)

    ## Tc の向上幅の平均値を計算
    #### all_True_boolsがTrueのサンプルのみを抽出でTcの差分を辞書データにいれる
    pred_tc = result_df.loc[result_df["all_True_bools"]]["pred_Tc"].values
    pred_tc = np.sort(pred_tc)[::-1]
    summary_dict["Tc_top5"] = np.mean(pred_tc[:5])
    summary_dict["Tc_top10"] = np.mean(pred_tc[:10])
    summary_dict["Tc_top30"] = np.mean(pred_tc[:30])
    summary_dict["Tc_top50"] = np.mean(pred_tc[:50])

    # 実験情報をいれる
    summary_dict["experiment_type"] = experiment_type
    summary_dict["method"] = method
    summary_dict.update(params_dict)

    result_csv_path = os.path.join(os.path.dirname(csv_path), "result_df.csv")
    result_df.to_csv(result_csv_path, index=False)
    summary_csv_path = os.path.join(os.path.dirname(csv_path), "summary.csv")
    pd.DataFrame([summary_dict]).to_csv(summary_csv_path, index=False)

    return summary_dict


# In[20]:


def search_best_threshold(total_generated_output, sg_model, element_table):
    """
    total_generated_output: 生成された組成式の出力
    sg_model: Tc予測モデル
    element_table: 元素のリスト
    """

    best_threshold = 0.0
    best_top30_tc_prediction_value = 0.0
    best_tc_predictions = np.nan
    for threshold in np.arange(0.01, 0.11, 0.01):
        atom_vectors = total_generated_output.clone()
        # 閾値以下の値を0にして、正規化
        atom_vectors[atom_vectors < threshold] = 0.0
        atom_vectors = torch.tensor(atom_vectors, dtype=torch.float32)
        atom_vectors = atom_vectors / atom_vectors.sum(dim=1, keepdim=True)
        # atom_vectors は96次元の元素ベクトルなので、Tc予測モデルに使うために118次元に拡張する
        dummy_vectors = torch.zeros(
            (atom_vectors.shape[0], 118 - atom_vectors.shape[1])
        )
        atom_vectors = torch.cat([atom_vectors, dummy_vectors], dim=1)
        # Tcを予測し、Top30の平均値を計算
        tc_values = sg_model(
            atom_vectors.view(-1, 118, 1, 1, 1).to(f"cuda")
        )  #:{gpu_device_number}'))
        tc_values = tc_values.cpu().detach().numpy().flatten()
        top30_tc_prediction_value = np.sort(tc_values)[::-1][:30].mean()

        # Top30の平均値が最大のthresholdを探す
        if top30_tc_prediction_value > best_top30_tc_prediction_value:
            best_top30_tc_prediction_value = top30_tc_prediction_value
            best_threshold = threshold
            best_tc_predictions = tc_values
        print(f"threshold: {threshold}, Top30の平均値: {top30_tc_prediction_value}")
    print(
        f"best_threshold: {best_threshold}, best_top30_tc_prediction_value: {best_top30_tc_prediction_value}"
    )

    return best_threshold, best_tc_predictions


# In[21]:


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


# # 評価

# In[22]:


import glob

from IPython.display import clear_output, display

# In[23]:


result_dir = "./results/hsc"

all_summaries = []
file_paths = glob.glob(os.path.join(result_dir, "**/*.npz"))
for p_i, npz_path in enumerate(file_paths):
    dummy_csv_path = npz_path.replace(".npz", ".csv")
    print(f"{p_i + 1}/{len(file_paths)}: {dummy_csv_path}")

    summary_dict = smact_etc_analysis_for_HSC(
        csv_path=dummy_csv_path,
        experiment_type="HSC",
        method="superdiff",
        Ef_criterion=0.0,
        elemnet=elemnet,
        sg_model=sg_model,
        params_dict={},
        hydrogen_thres=0.4,
        max_num_atoms=20,
        max_num_elements=3,
    )
    all_summaries.append(summary_dict)
    clear_output(wait=True)


# In[24]:


pd.DataFrame(all_summaries).to_csv(
    os.path.join(result_dir, "all_summaries.csv"), index=False
)


# In[25]:


df = pd.DataFrame(all_summaries)
# 数値型など、文字列・オブジェクト型以外のカラムを抽出
numeric_cols = df.select_dtypes(exclude=["object", "string"]).columns
# 'method' ごとに groupby し、抽出したカラムのみで集約処理を実施
df.groupby("method")[numeric_cols].mean()


# In[ ]:
