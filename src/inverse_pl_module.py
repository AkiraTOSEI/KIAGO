import os
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.nn.parameter import Parameter as Parameter
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.elemnet import ElemNet
from src.initializer import initialize_solution_candidates
from src.loss import IntegerLoss
from src.periodic_table import PeriodicTableSurrogateModel


class InvModule4PeriodicTable(pl.LightningModule):
    """pytorch lightning module for solving inverse problem"""

    def __init__(self, cfg: DictConfig, test: bool):
        super().__init__()
        self.cfg = cfg
        self.test = test

        # load Tc predeiction surrogate model
        self.sg_model = PeriodicTableSurrogateModel(cfg)
        self.sg_model.load_state_dict(
            torch.load(
                os.path.join(
                    cfg.output_dirs.sg_model_dir, cfg.sg_model.model_name + ".pt"
                )
            )
        )
        self.sg_model.eval()
        # load ElemNet to predict formation energy
        self.elemnet = ElemNet(
            86,
            "1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1",
            activation="relu",
            dropouts=[0.8, 0.9, 0.7, 0.8],
        )
        self.elemnet.load_state_dict(
            torch.load(
                os.path.join(cfg.output_dirs.sg_model_dir, cfg.sg_model.elemnet_path)
            )
        )
        self.elemnet.eval()
        self.ef_loss_coef = cfg.inverse_problem.loss.formation_energy_loss_coef
        ## Mask the atoms only used in ElemNet.
        self.elemnet_atom_ids = [
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
        self.elemnet_mask = torch.zeros(118).view(1, 118)
        self.elemnet_mask[:, self.elemnet_atom_ids] = 1.0

        self.type_of_atoms = cfg.general.type_of_atoms
        if self.cfg.inverse_problem.method == "NA":
            self.create_learnable_tensor()

        self.loss_func = self.construct_loss_func()
        self.batch_size = cfg.inverse_problem.method_parameters.iv_batch_size
        self.optimization_steps = (
            cfg.inverse_problem.method_parameters.optimization_steps
        )
        self.structure_list = cfg.inverse_problem.loss.structure_list
        self.opti_method = cfg.inverse_problem.method_parameters.optimizer
        self.lr = cfg.inverse_problem.method_parameters.iv_lr

        # loss
        self.structure_loss_fin_coef = (
            cfg.inverse_problem.loss.structure_loss_final_coef
        )
        self.structure_loss_srt_epoch = (
            cfg.inverse_problem.loss.structure_loss_srt_epoch
        )

        # constraint
        self.prepare_input_constraint()
        self.num_max_types = cfg.inverse_problem.constraint.num_max_types_of_atoms
        self.nmt_constraint_apply_step = (
            cfg.inverse_problem.constraint.num_max_types_of_atoms_constraint_apply_step
        )
        self.use_atom_mask = torch.Tensor([1.0])  # initialization

        # NeuLag導入までの一時的な処置
        if cfg.inverse_problem.loss.structure_loss_type == "normal":
            self.loss_calculator: nn.Module = IntegerLoss(
                structure_atom_count_range=cfg.inverse_problem.loss.structure_list,
                num_types_of_atoms=self.type_of_atoms,
            )
        else:
            raise Exception("integerize_loss_type error")

        # 解析用
        self.init_learnable_tensor = torch.clone(self.learnable_tensor)
        self.init_solution = (
            self.preprocess_x(self.init_learnable_tensor)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        self.init_tc = self.calculate_tc(self.init_learnable_tensor)
        self.init_ef = self.calculate_ef(self.init_learnable_tensor)

    def stuclture_loss_strongness(self):
        if self.structure_loss_srt_epoch > self.trainer.global_step:
            return 0.0
        else:
            return (
                self.structure_loss_fin_coef
                / (self.structure_loss_srt_epoch)
                * (self.trainer.global_step - self.structure_loss_srt_epoch + 1)
            )

    def create_learnable_tensor(self):
        """create leranable tensor for Neural Adjoint Method"""

        tensor_size = self.cfg.inverse_problem.method_parameters.iv_batch_size

        learnable_tensor = initialize_solution_candidates(
            self.cfg, tensor_size, self.test
        )
        learnable_tensor = torch.Tensor(
            learnable_tensor.reshape(tensor_size, self.type_of_atoms, 1, 1, 1)
        )
        self.initial_tensor = torch.clone(learnable_tensor)
        self.learnable_tensor = Parameter(learnable_tensor, requires_grad=True)
        self.cuv = Parameter(torch.eye(tensor_size), requires_grad=False)

    def construct_loss_func(
        self,
    ) -> Callable[
        [Tensor, Tensor, Tensor, Tensor, str], Tuple[Tensor, dict, Tensor, Tensor]
    ]:
        """
        define loss function
        """

        def loss_func(
            input: Tensor,
            y_tc_hat: Tensor,
            y_tc: Tensor,
            y_ef: Tensor,
            reduction: str = "mean",
        ) -> Tuple[Tensor, dict, Tensor, Tensor]:
            # tc_mae_loss = upperbounded_mae(y_tc_hat, y_tc)
            tc_mae_loss = -y_tc_hat.squeeze()
            ef_loss = y_ef.squeeze() * self.ef_loss_coef
            structure_loss, _ = self.loss_calculator.calculate_min_loss(input.squeeze())
            structure_loss *= self.stuclture_loss_strongness()

            loss_val = tc_mae_loss + ef_loss + structure_loss

            # NaN が発生した場合は 0 にする
            loss_val[torch.isnan(loss_val)] = 0.0

            if reduction == "mean":
                loss_val = loss_val.mean()
            elif reduction == "none":
                loss_val = loss_val
            else:
                raise Exception(" reduction is invalid. current value:", reduction)

            loss_dict = {
                "loss": loss_val.mean(),
                "mae_loss": tc_mae_loss.mean(),
                "ef_loss": ef_loss.mean(),
                "structure_loss": structure_loss.mean(),
            }
            return loss_val, loss_dict, tc_mae_loss, ef_loss

        return loss_func

    def calculate_tc(self, x):
        if self.sg_model.training:
            self.sg_model.eval()
        return self.sg_model(self.preprocess_x(x)).squeeze().detach().cpu().numpy()

    def calculate_ef(self, x):
        if self.elemnet.training:
            self.elemnet.eval()

        return (
            self.elemnet(
                self.preprocess_x(x)[
                    :, self.elemnet_mask.squeeze().to(torch.bool)
                ].squeeze()
            )
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

    def save_optimized_results(self):
        optimized_solutions = (
            self.preprocess_x(self.learnable_tensor).squeeze().detach().cpu().numpy()
        )
        init_solutions = self.init_solution
        optimized_tc = self.calculate_tc(self.learnable_tensor)
        optimized_ef = self.calculate_ef(self.learnable_tensor)

        if not os.path.exists(self.cfg.output_dirs.inverse_result_dir):
            os.makedirs(self.cfg.output_dirs.inverse_result_dir)

        np.savez(
            os.path.join(
                self.cfg.output_dirs.inverse_result_dir, "optimized_solutions.npz"
            ),
            optimized_solutions=optimized_solutions,
            optimized_tc=optimized_tc,
            optimized_ef=optimized_ef,
            init_solutions=init_solutions,
            init_tc=self.init_tc,
            init_ef=self.init_ef,
            random=np.random.default_rng(1234).random(20),
        )

    def prepare_input_constraint(self):
        # Create two constraint tensors. The first is the input constraint tensor that is not affected by optimization. The second is the tensor to narrow down the optimization target.
        self.input_constraint = self.cfg.inverse_problem.constraint.input_constraint
        if self.input_constraint is not None:
            print("USE input constraint:", self.input_constraint)
            self.constraint_sum = np.sum(self.input_constraint)
            assert self.constraint_sum < 1.0
            input_constraint_tensor = (
                torch.Tensor(self.input_constraint)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            self.input_constraint_tensor = nn.Parameter(
                input_constraint_tensor, requires_grad=False
            )

        else:
            self.constraint_sum = 0.0
            self.input_constraint_tensor = nn.Parameter(
                torch.Tensor([0.0]), requires_grad=False
            )

        self.optimization_mask = self.cfg.inverse_problem.constraint.optimization_mask
        if self.optimization_mask is not None:
            # optimization mask shape is (type_of_atoms,)
            self.optimization_mask = torch.Tensor(
                np.array(self.optimization_mask)
            ).unsqueeze(0)
            assert self.optimization_mask.shape[1] == self.type_of_atoms, "shape error"
            assert self.optimization_mask.shape[0] == 1, "shape error"
            # optimization mask contains 0 or 1. 0 means that the value is not optimized.
            assert torch.all(
                (self.optimization_mask == 0) | (self.optimization_mask == 1)
            )
        else:
            self.optimization_mask = torch.ones(1, self.type_of_atoms)

        self.optimization_mask = (
            self.optimization_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        self.optimization_mask_tesnor = nn.Parameter(
            self.optimization_mask, requires_grad=False
        )

    def normalization_x(self, x: Tensor) -> Tensor:
        """normalize x to be sum of 1 along with dim=1 after relu activation"""
        x = functional.relu(x)
        return x / x.sum(dim=1, keepdim=True)

    def apply_atom_mask_within_86(self, x: Tensor) -> Tensor:
        if x.device != self.elemnet_mask.device:
            self.elemnet_mask = self.elemnet_mask.to(x.device)
        while self.elemnet_mask.dim() < x.dim():
            self.elemnet_mask = self.elemnet_mask.unsqueeze(-1)
        x = x * self.elemnet_mask
        return x

    def apply_input_constraint(self, x: Tensor) -> Tensor:
        # apply optimization mask
        x = x * self.optimization_mask_tesnor
        x = self.apply_atom_mask_within_86(
            x
        )  # 86番目以降は0にする, ElemNetの入力には使わないことによる対応
        # normalization
        x = self.normalization_x(x)
        # apply input constraint
        x = (1.0 - self.constraint_sum) * x + self.input_constraint_tensor
        return x

    def apply_constraint_num_type_of_atoms(self, x: Tensor) -> Tensor:
        if x.device != self.use_atom_mask.device:
            self.use_atom_mask = self.use_atom_mask.to(x.device)
        x = x * self.use_atom_mask
        return self.normalization_x(x)

    def preprocess_x(self, x: Tensor) -> Tensor:
        """apply input constraint and normalization"""
        # x = self.normalization_x(x)
        x = self.apply_constraint_num_type_of_atoms(x)  # 元素数制約を適用
        x = self.apply_input_constraint(x)  # 原子種制約を適用
        return x

    def create_use_atom_mask(self):
        with torch.no_grad():
            x = self.preprocess_x(self.learnable_tensor)
        sorted_x, _ = torch.sort(x, dim=1, descending=True)
        thres = sorted_x[:, self.num_max_types : self.num_max_types + 1]
        self.use_atom_mask = (x - thres > 0).float()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        # return self.sg_model(x)

    def training_step(self, batch, batch_idx):
        if self.sg_model.training:
            self.sg_model.eval()
        if self.elemnet.training:
            self.elemnet.eval()
        y_tc_gt = batch
        if self.cfg.inverse_problem.method == "NA":
            x = self.preprocess_x(self.learnable_tensor)
            y_tc_hat = self.sg_model(x)
            y_ef_hat = self.elemnet(
                x[:, self.elemnet_mask.squeeze().to(torch.bool)].squeeze()
            )
            loss_each, metrics, tc_loss_each, ef_loss = self.loss_func(
                x, y_tc_hat, y_tc_gt, y_ef_hat, "none"
            )

        else:
            raise Exception("method error")

        self.y_gt = y_tc_gt
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        # sch = self.lr_schedulers()
        # sch.step(metrics["loss"])  # type: ignore

        return metrics

    def configure_optimizers(self):
        if self.opti_method == "Adam":
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            optimizer = torch.optim.Adam([self.learnable_tensor], lr=self.lr)

            self.use_sheduler = False
            return optimizer

        elif self.opti_method == "Adam-CA":
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr[0])
            optimizer = torch.optim.Adam([self.learnable_tensor], lr=self.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.lr[1])
            self.use_sheduler = "step"
            return [optimizer], [scheduler]

        else:
            raise Exception("optimizer error")

    def on_train_batch_end(self, out, batch, batch_idx):
        if (
            self.num_max_types > 1
            and self.trainer.global_step == self.nmt_constraint_apply_step
        ):
            self.create_use_atom_mask()
            print("<INFO>  create mask for num_types_of_atoms")

        if self.use_sheduler == "step":
            sch = self.lr_schedulers()
            sch.step()
