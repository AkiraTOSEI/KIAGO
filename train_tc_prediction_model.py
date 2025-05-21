import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.config import default_config
from src.surrogate_eval import MAE_PROB_SCORE_BY_TYPE
from src.surrogate_models import run_surrogate_experiment


def main_task_training(
    cfg: DictConfig,
    test_mode: bool = False,
):
    if test_mode:
        cfg.sg_model.max_epochs = 2

    result_dict, exp_name = run_surrogate_experiment(cfg, test_mode=test_mode)

    return result_dict, exp_name


def forward_main():
    test_mode = False

    map_types = [
        "AtomMap-base.npy",
    ]
    model_types = [
        "ResNet18",
    ]

    DATA_INFOs = [
        ["random", [0.05, 0.15]],  # ランダム分割での実験用
    ]
    element_division = True

    num_trial = 1
    effective_batch_size = 1024

    scores = []
    for trial in range(num_trial):
        for data_info in DATA_INFOs:
            for model_type, atom_map_type in zip(model_types, map_types):
                for optimizer, lr in zip(["Adam-CA"], [7e-5]):
                    batch_size = 1024

                    if effective_batch_size < batch_size:
                        continue
                    num_accum_batch = effective_batch_size // batch_size

                    if model_type.lower().startswith("resnet"):
                        optimizer = "Adam"
                    cfg = default_config()
                    cfg.sg_model.model_type = model_type
                    cfg.sg_model.atom_map_type = atom_map_type
                    # 基本的な設定
                    cfg.sg_model.batch_size = batch_size
                    cfg.sg_model.num_accum_batch = num_accum_batch

                    # optimizerの設定
                    cfg.sg_model.optimizer = optimizer
                    cfg.sg_model.lr = lr
                    cfg.sg_model.pretrain.optimizer = optimizer
                    cfg.sg_model.pretrain.lr = lr

                    # dataset settings
                    cfg.dataset.divide_method = data_info[0]
                    cfg.dataset.divide_infos = data_info[1]
                    cfg.dataset.element_division = element_division

                    """
                    main task training
                    """

                    result_dict, exp_name = main_task_training(cfg, test_mode)

                    d = np.load(
                        os.path.join(
                            "result",
                            cfg.output_dirs.surrogate_result_dir + ".npz",
                        )
                    )
                    scores.append(
                        MAE_PROB_SCORE_BY_TYPE(d).rename({"score": trial}, axis=1)
                    )

            pd.concat([score.set_index("sc_type") for score in scores], axis=1).to_csv(
                os.path.join("result", f"score_list__{exp_name}.csv")
            )


forward_main()
