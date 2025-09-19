#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[2]:


import os

import numpy as np

# # 関数など
# ## ElemNetのロード
# In[3]:
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


# In[4]:


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

# In[5]:


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


# In[6]:


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


# In[7]:


def Read_AtomMap(cfg: DictConfig):
    map_type = cfg.sg_model.atom_map_type
    if map_type.endswith("AtomMap-pg.npy") or map_type.endswith("AtomMap-base.npy"):
        atom_map = np.load(os.path.join(cfg.general.processed_data_dir, map_type))[
            np.newaxis, ...
        ]
    else:
        raise Exception(f"Invalid map_type: {map_type}")

    return atom_map


# In[8]:


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


# In[9]:


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


# In[10]:


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

# In[11]:


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


# In[ ]:


# In[ ]:


# In[12]:


# In[13]:


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


# In[14]:


import csv
import functools
import itertools
import math
from fractions import Fraction
from functools import reduce
from math import gcd

import numpy as np
import pandas
import smact
from pymatgen.core.composition import Composition
from smact.screening import pauling_test
from tqdm.notebook import tqdm


def reduce_by_gcd(stoichs):
    """
    Given a sequence of integer stoichiometric coefficients,
    find its GCD. If the GCD > 1, divide each element by it.
    Return the "reduced" stoichs.
    """
    g = reduce(gcd, stoichs)
    if g > 1:
        return tuple(x // g for x in stoichs)
    else:
        return tuple(stoichs)


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    """
    Define smact validation function. Basic Charge Neutrality and Electronegativity Balance checks.
    Same method as https://github.com/cseeg/DiSCoVeR-SuperCon-NOMAD-SMACT/blob/main/main.ipynb on SuperCon
    Adapted from CDVAE (https://arxiv.org/abs/2110.06197)
    """
    global compositions
    space = smact.element_dictionary(comp)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(comp)) == 1:
        return True, True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in comp]
        if all(is_metal_list):
            return True, True

    threshold = np.max(count)
    compositions = []

    total = 1
    for combo in ox_combos:
        total *= len(combo)
    # for ox_states in tqdm(itertools.product(*ox_combos),total=total,leave=False, desc=f'Validating Generated Compounds:{comp}'):
    is_neutral = False
    electroneg_OK = False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        # Electronegativity test
        if cn_e:
            is_neutral = True
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                break
    # print(compositions, is_neutral, comp, count, sum(count))
    return electroneg_OK, is_neutral


# In[15]:


def get_settings_of_element_substitution_in_proposed_method(npz_path):
    """
    npz_pathから、Element Substitutionの設定を取得する
    """
    remain_atom_dict = parse_formula(
        npz_path.split("/")[-2].split("___")[0].replace("__results", "").split("__")[1]
    )
    substitute_target, val_atom = (
        npz_path.split("/")[-2]
        .replace("__results", "")
        .split("__")[2]
        .replace("target-", "")
        .split("_ef_coef")[0]
        .split("-")
    )
    val_of_atom = float(val_atom)
    remain_atom_vector = np.array(
        create_atomic_vectors_from_formula_dict(remain_atom_dict, val_of_atom)
    )
    substitution_ox_state = int(
        npz_path.split("/")[-2].split("__")[0].replace("ox-", "")
    )
    return (
        remain_atom_dict,
        substitute_target,
        val_of_atom,
        remain_atom_vector,
        substitution_ox_state,
    )


# In[16]:


from typing import List


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
                atomic_str += f"{atomic_number_to_symbol[idx + 1]}_{{{str(np.round(count, 4))[:6]}}} "
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


def create_atomic_strings_for_smact_check(atomic_string):
    return atomic_string.replace(" ", "").replace("_{", "").replace("}", "")


# In[17]:


def atom_dict_to_smact_string(atom_dict):
    if type(atom_dict) is str:
        atom_dict = ast.literal_eval(atom_dict)
    elif type(atom_dict) is not dict:
        raise ValueError(f"atom_dict must be dict, but got {type(atom_dict)}")

    smact_string = ""
    for atom, num in atom_dict.items():
        smact_string += f"{atom}{num}"
    return smact_string


class integerize_atom_dict:
    def __init__(self, remain_atom_dict, val_of_atom, original_atom_dict, decimals=4):
        self.remain_atom_dict = remain_atom_dict.copy()
        self.val_of_atom = val_of_atom
        self.original_atom_dict = original_atom_dict.copy()
        self.decimals = decimals

    def integerize_dict(self, _atom_dict):
        atom_dict = _atom_dict.copy()

        # Unit cellのサイズを、remain_atom_dictの値から計算する
        for atom in self.remain_atom_dict.keys():
            num_unit_cell = atom_dict[atom] / self.remain_atom_dict[atom]
        check_array = np.array(
            [
                atom_dict[atom] / self.remain_atom_dict[atom]
                for atom in self.remain_atom_dict.keys()
            ]
        )
        # assert (np.abs(check_array - num_unit_cell) < 1E-4).all(), f"num_unit_cell is invalid. check_array={check_array}, num_unit_cell={num_unit_cell}"

        # Unit cell のサイズに合わせて、atom_dictの値を調整する
        atom_dict = {k: v / num_unit_cell for k, v in atom_dict.items()}

        # remain_atom_dictの値がうまく再現できているかを確認
        for key in self.remain_atom_dict.keys():
            assert abs(atom_dict[key] - self.remain_atom_dict[key]) < 1e-3, (
                f"atom_dict[{key}]={atom_dict[key]} is invalid"
            )

        # atom_dictから、remain_atom_dictの要素を削除する
        for key in self.remain_atom_dict.keys():
            atom_dict.pop(key)

        # Element substitutionの対象の原子のあるべき合計値を計算する
        sum_targeted_atoms_ideal = self.val_of_atom
        sum_targeted_atoms_real = sum(atom_dict.values())
        ratio = (
            1.0
            - (sum_targeted_atoms_real - sum_targeted_atoms_ideal)
            / sum_targeted_atoms_ideal
        )

        # まず、sumの値が整数になるように、全ての要素をratio倍する
        atom_dict = {k: v * ratio for k, v in atom_dict.items()}
        sum_value = sum(atom_dict.values())

        # 正確に小数点なしでroundするために、最後の要素のみ別処理
        sum_value = 0
        for i, key in enumerate(atom_dict.keys()):
            atom_value = atom_dict[key]
            if i == len(atom_dict.keys()) - 1:
                atom_dict[key] = float(
                    f"{np.round(sum_targeted_atoms_ideal - np.round(sum_value, self.decimals), self.decimals):.4f}"
                )
                # assert atom_dict[key] > 0, f"atom_dict[{key}]={atom_dict[key]} is invalid"
            else:
                atom_dict[key] = float(str(np.round(atom_value * ratio, self.decimals)))
                sum_value += float(f"{(np.round(atom_dict[key], self.decimals)):.4f}")
        for atom in self.remain_atom_dict.keys():
            atom_dict[atom] = float(f"{self.remain_atom_dict[atom]:.4f}")

        # 値がゼロになった元素（要素）を削除する
        atom_dict = {k: v for k, v in atom_dict.items() if v > 0}
        # atom_dict = convert_to_decimal(atom_dict)

        # assert sum(original_atom_dict.values())*num_unit_cell == sum(atom_dict.values()), f"sum(original_atom_dict.values())*num_unit_cell={sum(original_atom_dict.values())*num_unit_cell} is not equal to sum(atom_dict.values())={sum(atom_dict.values())}"
        # assert abs(sum(self.original_atom_dict.values())*num_unit_cell - sum(atom_dict.values()))<1E-6, atom_dict
        assert (
            sum(self.remain_atom_dict.values())
            + self.val_of_atom
            - sum(atom_dict.values())
            < 1e-8
        ), (
            "final check failes. sum(remain_atom_dict.values())+val_of_atom - sum(atom_dict.values())={sum(remain_atom_dict.values())+val_of_atom - sum(atom_dict.values())}"
        )
        return atom_dict


# In[18]:


def count_elements(formula_string):
    return len(formula_string.strip().split(" "))


# In[19]:


# import swifter


def create_smact_check_strings(df, remain_atom_dict, val_of_atom, original_atom_dict):
    """
    Element Substitution で、SMACT評価用のの組成を取得する。
    元素置換対象の元素の組成は丸められているので、合計しても元の組成にならない。そのため、元の組成に合わせるような丸め方をする
    """
    ## SMACT評価用のの組成を取得
    df["Comp_for_smact"] = df["Initial Optimized Composition"].apply(
        create_atomic_strings_for_smact_check
    )
    ## 原子比の小数点部分を、原子総数に合わせて丸める
    df["normalized_dict"] = (
        df["Comp_for_smact"]
        .swifter.apply(parse_formula)
        .swifter.apply(
            integerize_atom_dict(
                remain_atom_dict, val_of_atom, original_atom_dict
            ).integerize_dict
        )
    )
    df["composition_for_smact_check"] = df["normalized_dict"].apply(
        atom_dict_to_smact_string
    )

    # Elementの数をカウント
    df["num_elements"] = df["Initial Optimized Composition"].apply(count_elements)

    return df


# In[20]:


# filter_for_valid_generated_compounds関数をapplyで並列化する


class ElectroNegativity_Check:
    def __init__(self, NUM_MAX_ELEMENSTS):
        self.NUM_MAX_ELEMENSTS = NUM_MAX_ELEMENSTS

    def check(self, formula):
        """
        Return validity of the formula, elecneg_ok, neutral_ok
        """
        form_dict = Composition(formula).to_reduced_dict
        comp = tuple(form_dict.keys())
        count = list(form_dict.values())
        if len(comp) >= self.NUM_MAX_ELEMENSTS:
            return False, False, False

        # find least common multiple to get count as a tuple of ints
        denom_list = [(Fraction(x).limit_denominator()).denominator for x in count]
        lcm = functools.reduce(lambda a, b: a * b // math.gcd(a, b), denom_list)
        count_list = []
        for i in count:
            count_list.append(round(i * lcm))
        count = tuple(count_list)
        count = reduce_by_gcd(count)

        elecneg_ok, neutral_ok = smact_validity(comp, count)
        return True, elecneg_ok, neutral_ok


# In[21]:


def get_evaluate_data_of_element_substitution(npz_path, original_df, NUM_MAX_ELEMENSTS):
    """
    Element Substitutionの評価データを取得する
    """
    # ファイル名から、維持する組成と置換する元素とその酸化数を取得
    (
        remain_atom_dict,
        substitute_target,
        val_of_atom,
        remain_atom_vector,
        substitution_ox_state,
    ) = get_settings_of_element_substitution_in_proposed_method(npz_path)
    # 置換前の組成ベクトルを取得する。
    # ここはElement Substitutionで共通の解析のはず。
    original_atom_dict = remain_atom_dict.copy()
    original_atom_dict[substitute_target] = val_of_atom
    original_atom_vector = torch.tensor(
        np.array(create_atomic_vectors_from_formula_dict(original_atom_dict, 0.0)),
        dtype=torch.float32,
    ).to("cuda")
    # Referenceの組成式のSMACT評価
    _, reference_formulra_validation, _ = ElectroNegativity_Check(
        NUM_MAX_ELEMENSTS
    ).check(atom_dict_to_smact_string(original_atom_dict))

    # 時間がかかるので、計算対象をスクリーニング
    ## 置換の元素数や維持組成の原子数の合計が整数にならないものは除外する。また
    remain_sum_atoms = sum(remain_atom_dict.values())

    # SMACTでチェックできるような文字列を取得する。元素置換の場合はcreate_smact_check_strings関数で組成の端数を丸める。
    ## df['composition_for_smact_check'] にSMACTチェック用の組成式文字列が入る, num_elementsに元素数が入る
    original_df = create_smact_check_strings(
        original_df, remain_atom_dict, val_of_atom, original_atom_dict
    )

    print(f"remain_sum_atoms={remain_sum_atoms}, val_of_atom={val_of_atom}")
    if not (
        abs(round(remain_sum_atoms) - remain_sum_atoms) < 1e-5
        and abs(round(val_of_atom) - val_of_atom) < 1e-5
    ):
        # return None
        raise Exception("Please check the following path")
    ## SMACTがもともとOKでないものは除外する
    if not reference_formulra_validation:
        # return None
        raise Exception("Please check the following path")
    ## 水素化合物超伝導体は除外する
    if "H" in original_atom_dict.keys():
        # return None
        raise Exception("Please check the following path")

    other_data_dict = {
        "remain_atom_dict": remain_atom_dict,
        "substitute_target": substitute_target,
        "val_of_atom": val_of_atom,
        "substitution_ox_state": substitution_ox_state,
        "original_atom_dict": original_atom_dict,
        "remain_atom_vector": remain_atom_vector,
        "reference_formulra_validation": reference_formulra_validation,
    }

    return original_df, original_atom_vector, original_atom_dict, other_data_dict


# In[ ]:


# ## cform_superdiff.py

# In[22]:


import ast


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


# In[23]:


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


# In[24]:


import ast
import json
import os
import random

# In[25]:
import numpy as np
import pandas as pd
import smact
from pymatgen.core.periodic_table import Element as Element_pmg
from smact import Element as Element_smact


def random_partition(n_elems, n_atoms, rng, experiment_type):
    points = rng.uniform(0, 1, n_elems - 1)
    points = np.concatenate(([0], points, [1]))  # 0 と 1 を追加
    points *= n_atoms  # 原子数に変換
    if experiment_type == "ElementSubstitution":
        points = np.unique(np.round(points, 4))  # # pointsの重複を削除する
        points = points / n_atoms  # n_atomsで割る
    elif experiment_type == "Normal" or experiment_type == "HSC":
        assert np.mod(n_atoms, n_elems) == 0, (
            f"n_atoms({n_atoms}) must be divisible by n_elems({n_elems})"
        )
        points = np.unique(np.round(points))  # # pointsの重複を削除する
        points = np.round(points) / n_atoms  # n_atomsで整数になるように丸める
    else:
        raise Exception(f"Invalid experiment_type={experiment_type}")
    points = np.sort(points)  # 昇順にソート
    return points


# In[26]:


class dict_vector_converter:
    def __init__(self, all_atom_list):
        self.all_atom_list = all_atom_list

    def convert_dict_to_vector(self, elem_dict):
        """
        Describe: Convert element dictionary to atomic vector
        Args:
            elem_dict: dict, element dictionary
        Returns:
            atomic_vector: np.ndarray, atomic vector
        """
        atomic_vector = np.array(np.zeros(len(self.all_atom_list)), dtype=np.float32)
        for elem, num in elem_dict.items():
            atomic_vector[self.all_atom_list.index(elem)] = num
        # normalize
        atomic_vector /= np.sum(atomic_vector)
        return atomic_vector


# In[27]:


def find_divisors(n):
    candidates = np.arange(1, int(np.sqrt(n)) + 1)
    divisors = candidates[n % candidates == 0]  # n を割り切る数を取得
    paired_divisors = n // divisors  # 対応する割られる側の数を取得
    return np.unique(np.concatenate((divisors, paired_divisors)))  # 重複を除いてソート


# In[28]:


def get_random_divisor(n, rng, max_val):
    """
    ランダムなnの約数を得る
    """
    divisor_list = find_divisors(n)
    divisor_list = list(divisor_list[divisor_list < max_val])
    rng.shuffle(divisor_list)
    try:
        return divisor_list[0]
    except:
        print(n, divisor_list)
        raise Exception("No divisor is found")


# In[29]:


def sample_es_materials(
    experiment_type, exp_dict, oxidation_dict, rng, max_n_atoms, max_n_elems
):
    """
    評価のための材料をサンプリングする関数
    experiment_type=='Normal'の場合は、exp_dictの材料からランダムに元素を選んで、その元素を酸化数に応じてランダムな複数の元素と置換する
    experiment_type=='ElementSubstitution'の場合は、exp_dictの材料から指定した元素を指定した酸化数に応じた複数の元素と置換する
    Args:
        experiment_type: str, 実験の種類
        exp_dict: dict, 実験の情報
    Returns:
        remain_atom_dict: dict, 置換後の元素の辞書

    """

    if experiment_type == "Normal":
        # 元素置換する元素をランダムに選ぶ
        elem_list = ast.literal_eval(exp_dict["elems"])
        comp = ast.literal_eval(exp_dict["comp"])
        atom_dict = {elem: val for elem, val in zip(elem_list, comp)}
        original_atom_dict = atom_dict.copy()
        _bool = False
        for _ in range(100):
            rand_int = np.random.randint(0, len(elem_list))
            target_elem = elem_list[rand_int]
            if target_elem in smact.metals:
                _bool = True
                break
        assert _bool, "元素置換する元素が金属元素ではありません"
        # ランダムに選んだ元素から酸化数を選ぶ
        sustitute_target = target_elem
        val_of_atom = comp[elem_list.index(sustitute_target)]
        ox_states = Element_pmg(sustitute_target).common_oxidation_states
        substitution_ox_state = ox_states[rng.integers(0, len(ox_states))]

        # 選んだ元素を削除し、代わりにその酸化数に応じた元素を追加するための辞書を作成する
        atom_dict.pop(sustitute_target)
        remain_atom_dict = atom_dict.copy()
        new_atom_dict = atom_dict.copy()

    elif experiment_type == "ElementSubstitution":
        elem_list = ast.literal_eval(exp_dict["elems"])
        comp = ast.literal_eval(exp_dict["comp"])
        atom_dict = {elem: val for elem, val in zip(elem_list, comp)}
        original_atom_dict = atom_dict.copy()
        sustitute_target = exp_dict["sustitute_target"]
        val_of_atom = round(exp_dict["val_of_atom"])
        assert val_of_atom == exp_dict["val_of_atom"]
        substitution_ox_state = exp_dict["substitution_ox_state"]
        remain_atom_dict = atom_dict.copy()
        remain_atom_dict.pop(sustitute_target)
        new_atom_dict = remain_atom_dict.copy()

    elif experiment_type == "HSC":
        elem_list = ast.literal_eval(exp_dict["elems"])
        comp = ast.literal_eval(exp_dict["comp"])
        atom_dict = {elem: val for elem, val in zip(elem_list, comp)}
        original_atom_dict = atom_dict.copy()
        target_elem = elem_list.copy()
        target_elem.remove("H")
        sustitute_target = target_elem[0]
        val_of_atom = comp[elem_list.index(sustitute_target)]
        ox_states = Element_pmg(sustitute_target).common_oxidation_states
        substitution_ox_state = ox_states[rng.integers(0, len(ox_states))]

        # 選んだ元素を削除し、代わりにその酸化数に応じた元素を追加するための辞書を作成する
        atom_dict.pop(sustitute_target)
        remain_atom_dict = atom_dict.copy()
        new_atom_dict = atom_dict.copy()

    # 対象の元素の酸化数をベースにした元素リストを作成する
    ox_atom_list = oxidation_dict[substitution_ox_state].copy()

    if experiment_type == "ElementSubstitution":
        for atom in new_atom_dict.keys():
            if atom in ox_atom_list:
                ox_atom_list.pop(ox_atom_list.index(atom))

    # 元素置換は比率が非整数でもよい。組成ベースの方は整数でなければならない

    # 酸化数のリストからランダムな個数選んだ元素をランダムな比率で追加する
    ## ランダム数のランダムな元素の比率を得る
    if experiment_type == "Normal" or experiment_type == "HSC":
        if val_of_atom == round(val_of_atom):
            # val_of_atomが整数のときはランダムに拡大する倍率を選ぶ
            if max_n_atoms // sum(original_atom_dict.values()) > 1:
                n_cells = rng.integers(
                    1, max_n_atoms // sum(original_atom_dict.values())
                )  # ランダムに拡大する倍率を選ぶ
            else:
                n_cells = 1
        else:
            # val_of_atomが整数でないときは、val_of_atomが整数になるように拡大する倍率を選ぶ
            candidates = np.arange(1, max_n_atoms * 5)
            candidates = candidates[
                sum(original_atom_dict.values()) * candidates < max_n_atoms
            ]  # 元素の総量がmax_n_atomsを超えないようにする
            candidates = candidates[
                candidates * val_of_atom == np.floor(candidates * val_of_atom)
            ]  # val_of_atomが整数になるようにする
            rng.shuffle(candidates)
            n_cells = candidates[0]
        n_es_atoms = n_cells * val_of_atom  # 元素置換をする対象の元素の組成比を取得
        assert n_es_atoms == round(n_es_atoms), (
            f"n_es_atoms is not integer. n_es_atoms={n_es_atoms}"
        )
        n_es_atoms = round(n_es_atoms)
        n_es_elems = get_random_divisor(
            n_es_atoms,
            rng,
            max_val=min([len(ox_atom_list), max_n_elems - len(remain_atom_dict)]),
        )  # ランダムな種類の元素を,len(ox_atom_list))と10 - len(new_atom_dict)より小さい値で選ぶ
    elif experiment_type == "ElementSubstitution":
        n_cells = 1
        n_es_atoms = round(val_of_atom)
        n_es_elems = rng.integers(
            1, max_n_elems - len(new_atom_dict)
        )  # ランダム数の元素を選ぶ

    ratios = (
        random_partition(n_es_elems, n_es_atoms, rng, experiment_type) * val_of_atom
    )
    ratios = list(ratios[1:] - ratios[:-1])

    assert round(np.sum(ratios), 5) == round(val_of_atom, 5), (
        f"比率の合計が1になっていません. val_of_atom={val_of_atom}, sum(ratios)={np.sum(ratios)}"
    )
    ## ランダムな元素を選ぶ
    add_elems = np.array(ox_atom_list)[
        rng.choice(len(ox_atom_list), n_es_elems, replace=False)
    ]
    ## 選んだ元素をatom_dictに追加
    for elem, ratio in zip(add_elems, ratios):
        if elem in new_atom_dict.keys():
            new_atom_dict[elem] += ratio
        else:
            new_atom_dict[elem] = ratio
    # print(f"n_es_elems={n_es_elems}, n_cells={n_cells}, n_es_atoms={n_es_atoms}")
    # print(f"len(remain_atom_dict)={len(remain_atom_dict)}, len(new_atom_dict)={len(new_atom_dict)}")
    # print(f"len(add_elems)={len(add_elems)}, len(ratios)={len(ratios)}")
    # assert sum(new_atom_dict.values()) == sum(original_atom_dict.values()), '元素の総量が変わっています'
    # almost same であればよい
    assert np.allclose(
        sum(new_atom_dict.values()), sum(original_atom_dict.values()), atol=1e-3
    ), (
        f"元素の総量が変わっています. sum(new_atom_dict.values())={sum(new_atom_dict.values())}, sum(original_atom_dict.values())={sum(original_atom_dict.values())}"
    )

    # new_atom_dictをn_atoms倍する
    new_atom_dict = {k: v * n_cells for k, v in new_atom_dict.items()}
    assert round(n_cells * sum(original_atom_dict.values()), 4) == round(
        sum(new_atom_dict.values()), 4
    ), (
        f"元素の総量が変わっています. n_cells*sum(original_atom_dict.values()): {n_cells * sum(original_atom_dict.values())}, sum(new_atom_dict.values()): {sum(new_atom_dict.values())}"
    )
    assert max_n_atoms >= sum(new_atom_dict.values()), (
        "元素の総量がn_max_atomsを超えています"
    )

    # new_atom_dictをアルファベット順にソートする
    new_atom_dict = {k: new_atom_dict[k] for k in sorted(new_atom_dict.keys())}
    # print("n_atoms:", sum(new_atom_dict.values()), "n_types of atoms:", len(new_atom_dict.keys()))

    # new_atom_dictを文字列に変換する
    # atom_str = "".join([f"{k}{round(v,4)}" for k, v in new_atom_dict.items()])

    return new_atom_dict


# In[30]:


import datetime
import multiprocessing
import time

from joblib import Parallel, delayed


def smact_etc_analysis(
    csv_path,
    experiment_type,
    method,
    NUM_MAX_ELEMENSTS,
    Ef_criterion,
    elemnet,
    sg_model,
    elem_sub_esp=1e-5,
    exp_dict=None,
    original_df=None,
):
    """
    SMACTによるスクリーニングや、各評価指標の計算を行う
    """

    assert experiment_type in ["ElementSubstitution", "Normal", "HSC"]
    assert method in ["proposed", "SuperDiff", "heuristic"]

    # formation energy の損失coefを取得
    if method == "proposed" and experiment_type == "ElementSubstitution":
        assert exp_dict is None
        npz_path = csv_path.replace(".csv", ".npz")
        d = np.load(npz_path)
        original_df, original_atom_vector, original_atom_dict, other_data_dict = (
            get_evaluate_data_of_element_substitution(
                npz_path, pd.read_csv(csv_path), NUM_MAX_ELEMENSTS
            )
        )
        ef_coef = float(
            npz_path.split("/")[-2].split("___")[0].split("__results_ef_coef-")[-1]
        )
        method_params = npz_path.split("/")[-2].split("___")[-1]
        method_name = f"proposed---{method_params}"

    elif method == "proposed" and experiment_type == "Normal":
        npz_path = csv_path.replace(".csv", ".npz")
        d = np.load(npz_path)
        assert exp_dict is None
        original_df = pd.read_csv(csv_path)
        original_atom_dict = parse_formula(
            npz_path.split("/")[-2]
            .split("___")[0]
            .replace("__results_ef_coef", "")
            .split("-")[0]
        )
        original_df["composition_for_smact_check"] = original_df[
            "Rounded Optimized Composition"
        ].apply(create_atomic_strings_for_smact_check)
        original_df["normalized_dict"] = original_df[
            "composition_for_smact_check"
        ].apply(parse_formula)
        original_atom_vector = torch.tensor(
            np.array(create_atomic_vectors_from_formula_dict(original_atom_dict, 0.0)),
            dtype=torch.float32,
        ).to("cuda")
        ef_coef = float(
            npz_path.split("/")[-2].split("___")[0].split("__results_ef_coef-")[-1]
        )
        method_params = npz_path.split("/")[-2].split("___")[-1]
        method_name = f"proposed---{method_params}"

    elif method == "heuristic":
        assert original_df is not None and exp_dict is not None
        ef_coef = np.nan
        elem_list = ast.literal_eval(exp_dict["elems"])
        comp = ast.literal_eval(exp_dict["comp"])
        original_atom_dict = {elem: val for elem, val in zip(elem_list, comp)}
        original_atom_vector = torch.tensor(
            np.array(create_atomic_vectors_from_formula_dict(original_atom_dict, 0.0)),
            dtype=torch.float32,
        ).to("cuda")
        method_name = "heuristic"

        if experiment_type == "ElementSubstitution":
            remain_atom_dict = original_atom_dict.copy()
            remain_atom_dict.pop(exp_dict["sustitute_target"])
            remain_atom_vector = np.array(
                create_atomic_vectors_from_formula_dict(
                    remain_atom_dict, float(exp_dict["val_of_atom"])
                )
            )
            other_data_dict = {
                "remain_atom_dict": remain_atom_dict,
                "substitute_target": exp_dict["sustitute_target"],
                "val_of_atom": float(exp_dict["val_of_atom"]),
                "substitution_ox_state": int(exp_dict["substitution_ox_state"]),
                "original_atom_dict": original_atom_dict,
                "original_atom_vector": original_atom_vector,
                "remain_atom_vector": remain_atom_vector,
            }

    elif method == "SuperDiff":
        npz_path = csv_path.replace(".csv", ".npz")
        d = np.load(npz_path)
        ## experiment_typeを取得
        if npz_path.split("/")[-3] == "usual":
            experiment_type = "Normal"
            json_path = "./ref_mat/experiment_usual_dict.json"
        elif npz_path.split("/")[-3] == "element_substitution":
            experiment_type = "ElementSubstitution"
            json_path = "./ref_mat/experiment_element_substitution_dict.json"
        else:
            raise ValueError(f"Invalid experiment_type: {npz_path.split('/')[-3]}")
        ## 学習したモデルと, classificer guidanceの情報を取得
        guidance_and_model = npz_path.split("/")[-2]
        if guidance_and_model.startswith("without"):
            ef_coef = np.nan
            lr = np.nan
            model_type = guidance_and_model.split("__")[1]
            method_name = f"SuperDiff-{model_type}"
        elif guidance_and_model.startswith("with_"):
            ef_coef = 4.0  # 全実験で共通
            lr = float(guidance_and_model.split("__")[1].replace("lr-", ""))
            model_type = guidance_and_model.split("__")[2]
            method_name = f"SuperDiff-{model_type} w/ CD lr-{lr}"
        else:
            raise ValueError(f"Invalid guidance_and_model: {guidance_and_model}")
        ## ベースとなる組成を取得
        cform = npz_path.split("/")[-1].split("__")[1].replace(".npz", "")
        original_atom_dict = parse_formula(cform)
        original_atom_dict = {
            k: original_atom_dict[k] for k in sorted(original_atom_dict.keys())
        }  # sort
        original_atom_vector = torch.tensor(
            np.array(create_atomic_vectors_from_formula_dict(original_atom_dict, 0.0)),
            dtype=torch.float32,
        ).to("cuda")
        ## 元素置換の場合、実験情報を取得する
        with open(json_path, "r") as f:
            experiment_dict_list = json.load(f)
        if experiment_type == "ElementSubstitution":
            elems = list(original_atom_dict.keys())
            elems.sort()
            flg = False
            for exp_dict in experiment_dict_list:
                if exp_dict["elems"] == str(elems):
                    flg = True
                    break
            assert flg, f"exp_dict is not found. elems={elems}"
            remain_atom_dict = ast.literal_eval(exp_dict["remain_atom_dict"])
            val_of_atom = float(exp_dict["val_of_atom"])
            remain_atom_vector = np.array(
                create_atomic_vectors_from_formula_dict(remain_atom_dict, val_of_atom)
            )
            other_data_dict = {
                "remain_atom_dict": remain_atom_dict,
                "substitute_target": exp_dict["sustitute_target"],
                "val_of_atom": float(exp_dict["val_of_atom"]),
                "substitution_ox_state": int(exp_dict["substitution_ox_state"]),
                "original_atom_dict": original_atom_dict,
                "original_atom_vector": original_atom_vector,
                "remain_atom_vector": remain_atom_vector,
            }

        total_generated_output = torch.tensor(d["total_generated_output"]).squeeze()
        ## npzファイルの元素ベクトルを使って、まず閾値を決めたあとに、SMACTのチェックを行う
        best_threshold, best_tc_predictions = search_best_threshold(
            total_generated_output, sg_model, element_table
        )
        ## 分布を得たいため、スクリーニング前のTc, Efを取得
        all_atom_vector = total_generated_output.squeeze()
        all_atom_vector[all_atom_vector < best_threshold] = 0.0  # 閾値を反映
        normed_all_atom_vector = all_atom_vector / all_atom_vector.sum(
            dim=1, keepdim=True
        )
        all_atom_vector_for_tc = torch.concat(
            [
                normed_all_atom_vector,
                torch.zeros(
                    (
                        normed_all_atom_vector.shape[0],
                        118 - normed_all_atom_vector.shape[1],
                    ),
                    device=normed_all_atom_vector.device,
                ),
            ],
            dim=1,
        ).float()
        all_atom_vector_for_ef = normed_all_atom_vector[:, :86].float()
        with torch.no_grad():
            all_tc = (
                (sg_model(all_atom_vector_for_tc.view(-1, 118, 1, 1, 1).to("cuda")))
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
            all_ef = (
                (elemnet(all_atom_vector_for_ef.to("cuda")))
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        ## ElemNet適用のため86番目以降の要素がすべて0のものを取得
        all_atom_vector = all_atom_vector[all_atom_vector[:, 86:].sum(dim=1) == 0.0]
        ## 得られたベクトルを元に、SMACTのチェックを行うための文字列にする
        original_df = pd.DataFrame(
            Atom_vec2string_converter().vector2string(
                all_atom_vector.detach().cpu().numpy()
            )
        )
        original_df[0] = original_df[0].apply(create_atomic_strings_for_smact_check)
        original_df.rename(columns={0: "composition_for_smact_check"}, inplace=True)
        original_df["normalized_dict"] = original_df[
            "composition_for_smact_check"
        ].apply(parse_formula)

    else:
        raise ValueError(f"Invalid method: {method}")

    # SMACTのチェック
    ## 一意の組成のみを取得
    unique_df = original_df.drop_duplicates(
        subset=["composition_for_smact_check"]
    ).copy()  # .reset_index()
    original_df["unique_composition_bools"] = ~original_df[
        "composition_for_smact_check"
    ].duplicated(keep="first")

    ## NUM_MAX_ELEMENSTS元素数より小さい組成のみを抽出
    # unique_df['valid_sample_bool'] = unique_df['num_elements'] < NUM_MAX_ELEMENSTS

    ## 並列化したSMACTチェック処理を実行
    print(f"Start Smact Check. current date and time: {datetime.datetime.now()}")
    srt_time = time.time()
    ## 並列化して実行
    checker = ElectroNegativity_Check(NUM_MAX_ELEMENSTS)
    n_jobs = min(multiprocessing.cpu_count(), 8)
    results = Parallel(n_jobs=n_jobs)(
        delayed(checker.check)(formula)
        for formula in unique_df["composition_for_smact_check"]
    )
    ## 結果を展開して DataFrame に代入
    unique_df["valid_sample_bool"], unique_df["elecneg_ok"], unique_df["neutral_ok"] = (
        zip(*results)
    )
    eps_time = time.time() - srt_time
    print(
        f"-> End Smact Check. current date and time: {datetime.datetime.now()}, elapsed time (hour:min:sec ): {str(datetime.timedelta(seconds=eps_time))}"
    )

    ## base材料のTcを予測
    input_vec = original_atom_vector.view(1, -1, 1, 1, 1)
    print("original_atom_vector.shape", original_atom_vector.shape)
    print("input_vec.shape", input_vec.shape)
    base_pred_Tc = (sg_model(input_vec)).detach().cpu().numpy().flatten()[0]
    ### normalizeした組成のTcを予測
    #### df['normalized_dict'] に対しcreate_atomic_vectors_from_formula_dictを適用してベクトル化し、Tcを予測
    atom_vectors = (
        unique_df["normalized_dict"]
        .apply(lambda x: np.array(create_atomic_vectors_from_formula_dict(x, 0.0)))
        .values
    )
    atom_vectors = torch.tensor(np.stack(atom_vectors), dtype=torch.float32).to("cuda")
    normalized_pred_Tc = (
        sg_model(atom_vectors.view(len(atom_vectors), 118, 1, 1, 1))
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
    normalized_pred_Ef = elemnet(atom_vectors[:, :86]).detach().cpu().numpy().flatten()

    result_df_dict = {
        "formula": unique_df.composition_for_smact_check.values,
        "atom_dict": unique_df.normalized_dict.values,
        "valid_sample_bools": unique_df.valid_sample_bool.values,
        "pred_Tc": normalized_pred_Tc,
        "pred_ef": normalized_pred_Ef,
        "Ef_bools": normalized_pred_Ef < Ef_criterion,
        "elecneg_bools": unique_df.elecneg_ok.values,
        "neutral_bools": unique_df.neutral_ok.values,
        "formula": unique_df.composition_for_smact_check.values,
    }

    # Element substitutionの評価
    if experiment_type == "ElementSubstitution":
        remain_atom_vector = other_data_dict["remain_atom_vector"]
        remain_atom_dict = other_data_dict["remain_atom_dict"]

        ### 保持する元素が保持されているかどうか
        subtracted_vector = np.abs(
            atom_vectors.detach().cpu().numpy() - remain_atom_vector
        )
        remain_atom_index = [
            define_atom_list().index(key) for key in remain_atom_dict.keys()
        ]
        Target_remain_bools = (
            subtracted_vector[:, remain_atom_index].sum(axis=1) < elem_sub_esp
        )
        ### Element Sustitutionの元素が合計量が正しいかどうか
        val_subsutitution = 1.0 - np.sum(remain_atom_vector)
        subtracted_vector = atom_vectors.detach().cpu().numpy().sum(
            axis=1
        ) - atom_vectors.detach().cpu().numpy()[:, remain_atom_index].sum(axis=1)
        Non_Target_remain_bools = (
            np.abs(subtracted_vector - val_subsutitution) < elem_sub_esp
        )
        ### 結果をまとめる
        result_df_dict["Target_remain_bools"] = Target_remain_bools
        result_df_dict["Non_Target_remain_bools"] = Non_Target_remain_bools

    # 結果をまとめる
    result_df = pd.DataFrame(result_df_dict)
    ## result_df_dict.keys()のなかで、boolsが含まれるものを取得し、すべてのboolsがTrueのサンプルのみを抽出
    bool_keys = [key for key in result_df_dict.keys() if key.endswith("bools")]
    result_df["all_True_bools"] = result_df[bool_keys].all(axis=1)
    result_df["base_pred_Tc"] = base_pred_Tc

    # 統計量をdictにまとめる
    sorted_original_atom_dict = {
        k: original_atom_dict[k] for k in sorted(original_atom_dict.keys())
    }
    summary_dict = {
        "base_atom_dict": sorted_original_atom_dict,
        "ef_coef": ef_coef,
        "num_total_samples": len(original_df),
        "unique_rate": len(unique_df) / len(original_df),
        "base_pred_Tc": base_pred_Tc,
        "valid_rate": unique_df["valid_sample_bool"].sum() / len(unique_df),
        "neutral_rate (in valid)": unique_df["neutral_ok"].sum()
        / unique_df["valid_sample_bool"].sum(),
        "elecneg_rate (in valid)": unique_df["elecneg_ok"].sum()
        / unique_df["valid_sample_bool"].sum(),
    }

    if experiment_type == "ElementSubstitution":
        summary_dict["remain_atom_dict"] = remain_atom_dict
        summary_dict["substitute_target"] = other_data_dict["substitute_target"]
        summary_dict["val_of_atom"] = other_data_dict["val_of_atom"]
        summary_dict["substitution_ox_state"] = other_data_dict["substitution_ox_state"]

    # 実験手法の詳細を追加する
    summary_dict["experiment_type"] = experiment_type
    summary_dict["method"] = method_name
    summary_dict["ef_coef"] = ef_coef

    # すべてのboolsの達成率を計算
    for b_key in bool_keys + ["all_True_bools"]:
        summary_dict[f"{b_key}_rate"] = result_df[b_key].sum() / len(result_df)

    ## Tc の向上幅の平均値を計算
    #### all_True_boolsがTrueのサンプルのみを抽出でTcの差分を辞書データにいれる
    result_df["delta_tc"] = result_df["pred_Tc"] - base_pred_Tc
    delta_tc = (
        result_df.loc[result_df["all_True_bools"]]["pred_Tc"] - base_pred_Tc
    ).values
    delta_tc = np.sort(delta_tc)[::-1]
    summary_dict["ΔTc_top10"] = np.mean(delta_tc[:10])
    summary_dict["ΔTc_top30"] = np.mean(delta_tc[:30])
    summary_dict["ΔTc_top50"] = np.mean(delta_tc[:50])

    # delta Tcが高い順にソートし、最適化した組成とTcをいくつか取得する
    sorted_df = result_df.loc[result_df["all_True_bools"]].sort_values(
        "delta_tc", ascending=False
    )
    ## 10個のサンプルを取得. それぞれformulaとTcを合わせた文字列形式で取得する
    top10_df = sorted_df.head(10)
    top10_df["result_string"] = top10_df["formula"].fillna("").astype(str) + top10_df[
        "pred_Tc"
    ].apply(lambda x: f" ({x:.2f} K)")
    result_string = ""
    for i, row in top10_df.iterrows():
        result_string += f"{row['result_string']} \n"
    summary_dict["top10_result_string"] = result_string

    result_csv_path = csv_path.replace("optimized_solutions.csv", "result_df.csv")
    result_df.to_csv(result_csv_path)

    return summary_dict


# In[31]:


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


from itertools import product


# oxidation_statesのN個の組み合わせを求める
def possible_sum_oxidaion_states_of_N_of_Element(N, elem):
    oxidation_states = smact.Element(elem).oxidation_states
    if len(oxidation_states) == 0:
        return []
    return list(
        set(np.sum(np.array(list(product(oxidation_states, repeat=N))), axis=1))
    )


def possible_sum_oxidaion_states_of_N_of_Hydrogen(N):
    # 各変数  a_i ​ が -1または +1 の値を取り、 N 個あるとすると、1 の数を x とすると全体の和 S は x - (N - x) = 2x - N となる。
    # この S は -N から N までの2ずつ増加する値を取る。
    return list(range(-N, N + 1, 2))


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


class check_HSC_heutrality:
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


class check_hydrogen_ratio:
    def __init__(self, hydrogen_thres):
        self.hydrogen_thres = hydrogen_thres

    def check(self, formula):
        elements, comps = get_list_data_from_formula_string(formula)
        if "H" in elements:
            return self.hydrogen_thres <= comps[elements.index("H")] / sum(comps) < 1.0
        else:
            return False


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


# In[32]:


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
    original_df=None,
    exp_dict=None,
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
    elif method == "heuristic":
        assert original_df is not None and exp_dict is not None
        ef_coef = np.nan
        elem_list = ast.literal_eval(exp_dict["elems"])
        comp = ast.literal_eval(exp_dict["comp"])
        original_atom_dict = {elem: val for elem, val in zip(elem_list, comp)}
        original_atom_vector = torch.tensor(
            np.array(create_atomic_vectors_from_formula_dict(original_atom_dict, 0.0)),
            dtype=torch.float32,
        ).to("cuda")
        method = "heuristic"

    # 一意のもののみを取得する
    unique_df = original_df.drop_duplicates(subset=["neutral_check_formula"]).copy()
    # 電気的中性のチェっっく
    check_neutrality_from_formula_string = check_HSC_heutrality(
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

    result_csv_path = csv_path.replace(".csv", "_result_df.csv")
    result_df.to_csv(result_csv_path, index=False)
    summary_csv_path = csv_path.replace(".csv", "_summary.csv")
    pd.DataFrame([summary_dict]).to_csv(summary_csv_path, index=False)

    return summary_dict


# In[33]:


from IPython.display import Math, clear_output


# 精製物のサンプルを取得
def display_samples(df):
    tmp_df = df.reset_index()
    for i, row in tmp_df.iterrows():
        display(row.method + ":" + str(row.base_atom_dict))
        if type(row["top10_result_string"]) is not str:
            continue
        for sample_i, sample_str in enumerate(row["top10_result_string"].split("\n")):
            if sample_str == "":
                continue
            formula, tc = sample_str.split(" (")
            atom_dict = parse_formula(formula)
            latex_formula = "\mathrm{"
            for key, value in atom_dict.items():
                latex_formula += f"{key}_{{{(value):.3f}}}"
            latex_formula += "}  (" + tc
            display(Math(latex_formula))
            if sample_i > 5:
                break
        print("")


# In[34]:


def formula_to_dict_style_formula(formula):
    mate_str = ""
    for key, value in parse_formula(formula).items():
        mate_str += f"{key}_" + "{" + f"{value}" + "} "
    mate_str = mate_str.strip()
    return mate_str


# # 評価結果

# In[35]:


from IPython.display import clear_output, display

# In[36]:


rng = np.random.default_rng(1234)
oxidation_dict = create_oxidation_elem_dict()


# In[39]:


# 合計で45 minくらいかかる
for experiment_type in ["Normal", "ElementSubstitution", "HSC"]:
    # json fileのよみこみ
    if experiment_type == "Normal":
        json_path = "./ref_mat/experiment_usual_dict.json"
        fname = "normal"
        max_n_atoms, max_n_elems = 200, 10

    elif experiment_type == "ElementSubstitution":
        json_path = "./ref_mat/experiment_element_substitution_dict.json"
        fname = "es"
        max_n_atoms, max_n_elems = 200, 10

    elif experiment_type == "HSC":
        json_path = "./ref_mat/experiment_hsc_dict.json"
        fname = "hsc"
        max_n_atoms, max_n_elems = 15, 3

    else:
        raise ValueError(f"Invalid experiment_type: {experiment_type}")

    all_summary_dicts = []
    with open(json_path) as f:
        experiment_dict_list = json.load(f)

    for i, exp_dict in enumerate(experiment_dict_list):
        if exp_dict["elems"] == "['Cs', 'Fe', 'Se', 'Tl']":
            # この条件だけは、うまく元素置換できないのでスキップ
            continue
        elif exp_dict["elems"] == "['H', 'Th']":
            # この条件は、数が原子数が多いため、スキップ
            continue

        if exp_dict["elems"] == "['Bi', 'Ca', 'Cu', 'Fe', 'O', 'Sr']":
            continue

        print(
            f"{i + 1}/{len(experiment_dict_list)}, experiment_type: {experiment_type}"
        )
        result_dir = os.path.join("rule_based_results", experiment_type)
        os.makedirs(result_dir, exist_ok=True)
        elem_list = ast.literal_eval(exp_dict["elems"])
        comp = ast.literal_eval(exp_dict["comp"])
        atom_str = "".join([f"{k}{round(v, 4)}" for k, v in zip(elem_list, comp)])
        csv_path = os.path.join(result_dir, f"{atom_str}.csv")
        # ヒューリスティック手法で、元素置換の評価データを取得する
        dict_converter = dict_vector_converter(define_atom_list())
        atom_str_vector = np.array(
            [
                sample_es_materials(
                    experiment_type,
                    exp_dict,
                    oxidation_dict,
                    rng,
                    max_n_atoms=max_n_atoms,
                    max_n_elems=max_n_elems,
                )
                for _ in range(2048 * 2)
            ]
        )
        original_df = pd.DataFrame(
            {
                "composition_for_smact_check": atom_str_vector,
                "normalized_dict": atom_str_vector,
            }
        )
        original_df["composition_for_smact_check"] = original_df[
            "normalized_dict"
        ].apply(lambda x: "".join([f"{k}{round(v, 4)}" for k, v in x.items()]))
        if experiment_type == "ElementSubstitution" or experiment_type == "Normal":
            summary_dict = smact_etc_analysis(
                csv_path,
                experiment_type,
                method="heuristic",
                NUM_MAX_ELEMENSTS=10,
                Ef_criterion=0.0,
                elemnet=elemnet,
                sg_model=sg_model,
                elem_sub_esp=1e-5,
                exp_dict=exp_dict,
                original_df=original_df,
            )
        elif experiment_type == "HSC":
            hydrogen_thres = 0.4
            original_df.to_csv("test.csv")
            original_df["neutral_check_formula"] = original_df[
                "composition_for_smact_check"
            ].apply(formula_to_dict_style_formula)
            summary_dict = smact_etc_analysis_for_HSC(
                csv_path,
                experiment_type,
                method="heuristic",
                Ef_criterion=0.0,
                elemnet=elemnet,
                sg_model=sg_model,
                params_dict={},
                hydrogen_thres=hydrogen_thres,
                max_num_atoms=max_n_atoms,
                max_num_elements=max_n_elems,
                original_df=original_df,
                exp_dict=exp_dict,
            )
        else:
            raise ValueError(f"Invalid experiment_type: {experiment_type}")
        clear_output()
        all_summary_dicts.append(summary_dict)

    pd.DataFrame(all_summary_dicts).to_csv(
        f"{result_dir}/{fname}_summary.csv", index=False
    )


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[40]:


# In[41]:
import glob
import os

import numpy as np
import pandas as pd
from IPython.display import clear_output, display

for path in glob.glob(f"./rule_based_results/*/*_summary.csv"):
    if os.path.basename(path).startswith("H"):
        continue
    print(path.split("/")[-1])
    display(pd.read_csv(path))
