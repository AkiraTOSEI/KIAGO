import os
import shutil
from typing import Any, Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from src.forward_p_table_datamodules import LitModel, PeriodicTableDataModule
from src.surrogate_eval import MAE_PROB_SCORE_BY_TYPE
from src.utils import cfg_add_infos


def predict_and_save(
    trainer: Trainer,
    model: nn.Module,
    data_module: Any,
    exp_name: str,
    include_train: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Function to predict and save results.

    Args:
        trainer (pytorch_lightning.Trainer): The trainer.
        model (torch.nn.Module): The model.
        data_module (Any): The data module.
        exp_name (str): The experiment name.

    Returns:
        Dict[str, np.ndarray]: The result dictionary containing prediction results and loss.
    """
    # prediction result for analysis
    mae_func = torch.nn.L1Loss()
    prediction_results = trainer.predict(model, datamodule=data_module)
    data_result_dict = dict()
    result_dict = dict()

    if include_train:
        phase_list = ["train", "valid", "test"]
    else:
        phase_list = ["valid", "test"]

    if not len(prediction_results) == len(phase_list):
        raise ValueError("The number of prediction results is not correct.")

    for phase, results in zip(phase_list, prediction_results):
        data_result_dict[f"{phase}_x"] = (
            torch.cat([x[0] for x in results], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        data_result_dict[f"{phase}_y_gt"] = (
            torch.cat([x[1] for x in results], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        data_result_dict[f"{phase}_y_hat"] = (
            torch.cat([x[2] for x in results], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        data_result_dict[f"{phase}_year"] = (
            torch.cat([x[3] for x in results], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # initialize the result directory
    result_dir = os.path.join("result")
    os.makedirs(result_dir, exist_ok=True)

    # save the results
    if include_train:
        np.savez(
            os.path.join(result_dir, exp_name + ".npz"),
            train_y_gt=data_result_dict["train_y_gt"],
            train_y_hat=data_result_dict["train_y_hat"],
            train_year=data_result_dict["train_year"],
            valid_x=data_result_dict["valid_x"],
            valid_y_gt=data_result_dict["valid_y_gt"],
            valid_y_hat=data_result_dict["valid_y_hat"],
            valid_year=data_result_dict["valid_year"],
            test_x=data_result_dict["test_x"],
            test_y_gt=data_result_dict["test_y_gt"],
            test_y_hat=data_result_dict["test_y_hat"],
            test_year=data_result_dict["test_year"],
        )
    else:
        np.savez(
            os.path.join(result_dir, exp_name + ".npz"),
            valid_x=data_result_dict["valid_x"],
            valid_y_gt=data_result_dict["valid_y_gt"],
            valid_y_hat=data_result_dict["valid_y_hat"],
            valid_year=data_result_dict["valid_year"],
            test_x=data_result_dict["test_x"],
            test_y_gt=data_result_dict["test_y_gt"],
            test_y_hat=data_result_dict["test_y_hat"],
            test_year=data_result_dict["test_year"],
        )

    return result_dict


def run_surrogate_experiment(
    cfg: DictConfig,
    skip_mode: bool = False,
    test_mode: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Function to run an experiment.

    Args:
        args (Dict[str, Any]): Dictionary with arguments (divide_method, divide_infos, train_balancing, batch_size, learning_rate).
        processed_data_dir (str): Directory with processed data.
        skip_mode (bool): Skip mode flag.
        test_mode (bool): Test mode flag.

    Returns:
        result_dict (Dict[str, Any]): Dictionary with results.
        exp_name (str): Experiment name.
    """
    # set up experiment dictionary
    cfg_add_infos(cfg, test_mode)
    exp_name = cfg.experiment_names.surrogate
    print("-----------------------------------")
    print("Experiment name: ", exp_name)
    print("-----------------------------------")

    # initialize the logger
    logdir = f"logs/{exp_name}"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    tb_logger = TensorBoardLogger(logdir)

    """
    # predict_and_save で使うのであとで再実装する
    if skip_mode and not test_mode:
        print("Skipping experiment: ", exp_name)
        print("\n \n \n \n \n")
        return {}, exp_name
    """

    # initialize the data module
    data_module = PeriodicTableDataModule(
        cfg=cfg,
        test_mode=test_mode,
    )

    # initialize the model
    model = LitModel(cfg)

    # initialize the trainer
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=50)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # deepspeed
    if cfg.general.devices > 1:
        # assert cfg.general.deep_speed is not None
        strategy = cfg.general.deep_speed
        precision = 16

    if cfg.general.deep_speed is not None:
        strategy = cfg.general.deep_speed
        precision = 16
    else:
        strategy = None
        precision = 32

    trainer = Trainer(
        max_epochs=cfg.sg_model.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        devices=1,
        accelerator="gpu",
        accumulate_grad_batches=cfg.sg_model.num_accum_batch,
        profiler="pytorch",
        strategy=strategy,
        precision=precision,
        enable_progress_bar="notebooks" in os.getcwd(),
    )

    # train the model
    trainer.fit(model, datamodule=data_module)

    # test the model
    result_dict = predict_and_save(
        trainer, model, data_module, exp_name, include_train=False
    )

    # model save
    if not test_mode:
        os.makedirs("surrogate_models", exist_ok=True)
        torch.save(model.sg_model.state_dict(), f"surrogate_models/{exp_name}.pt")

    return result_dict, exp_name
