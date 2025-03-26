import os
import shutil
import time
from typing import Optional

import pytorch_lightning as pl
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .inverse_datamodules import InverseDataModule4PariodicTable
from .inverse_pl_module import InvModule4PeriodicTable
from .visualize_solution import display_optimized_solutions


class ForwardModelFreezeCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        for param in pl_module.sg_model.parameters():
            param.requires_grad = False
        pl_module.sg_model.eval()
        for param in pl_module.elemnet.parameters():
            param.requires_grad = False
        pl_module.elemnet.eval()


def InverseOpt4PeriodicTable(cfg: DictConfig, test_mode: bool = False):
    """
    inverse optimization for both Neural Lagrangian method and NA method.
    Args:
      cfg(DictConfig) config
      test(bool) : whether to use test-run-mode
    """

    # initial setting from config
    opt_steps = cfg.inverse_problem.method_parameters.optimization_steps

    # candidate reduction settings
    if cfg.inverse_problem.method == "NeuralLagrangian":
        reduction_schedule = cfg.inverse_problem.candidate.reduction_schedule
        print("<info> reduction schedule:", reduction_schedule)
        if len(reduction_schedule) == 1 and reduction_schedule[0] == 1:
            cfg.inverse_problem.candidate.use_candidate_selection = False
            max_epochs = 1
            cfg.inverse_problem.method_parameters.optimization_steps = opt_steps
        else:
            cfg.inverse_problem.candidate.use_candidate_selection = True
            cfg.inverse_problem.method_parameters.optimization_steps = opt_steps // len(
                reduction_schedule
            )
            max_epochs = len(reduction_schedule)
    elif cfg.inverse_problem.method == "NA":
        max_epochs = 1
    else:
        raise Exception("method name error")

    # if test mode, optimization steps become smaller
    if test_mode:
        cfg.inverse_problem.method_parameters.optimization_steps = 5

    # initialization
    iv_model = InvModule4PeriodicTable(cfg, test_mode)
    datamodule = InverseDataModule4PariodicTable(cfg, test_mode)

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # initialize the logger
    logdir = f"logs/test"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs("logs/test", exist_ok=True)
    tb_logger = TensorBoardLogger(logdir)

    # deepspeed setting
    if cfg.general.devices > 1:
        # assert cfg.general.deep_speed is not None
        strategy = cfg.general.deep_speed
        precision = 16

    if cfg.general.deep_speed is not None and not "notebooks" in os.getcwd():
        strategy = cfg.general.deep_speed
        precision = 32
    else:
        strategy = None
        precision = 32

    # inverse problem optimization training and get results
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=tb_logger,
        accelerator="gpu",
        devices=1,
        # gradient_clip_val=cfg.inverse_problem.method_parameters.grad_clip,
        callbacks=[ForwardModelFreezeCallback(), lr_monitor],
        precision=precision,
        strategy=strategy,
        enable_progress_bar="notebooks" in os.getcwd(),
    )
    trainer.fit(iv_model, datamodule=datamodule)

    # save optimization results
    iv_model.save_optimized_results()

    # plot optimized solutions
    top_N_Tc_mean = display_optimized_solutions(cfg, iv_model)

    return iv_model, top_N_Tc_mean
