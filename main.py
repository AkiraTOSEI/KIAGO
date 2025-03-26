import json
import os

import numpy as np
import pandas as pd

from src.main_experiment import main_experiment

# 通常の実験

test_mode = False
for experiment_type in ["HSC", "Normal", "ElementSubstitution"]:
    # json fileのよみこみ
    if experiment_type == "Normal":
        json_path = (
            "reference_sample_configs/experiment_usual_dict.json"
        )
        param_dict_list = [
            {
                "lr": 0.03,
                "mutate_prob": 0.29,
                "perturb_prob": 0.22,
                "perturb_val": 0.03,
                "max_num_atoms": 200,
            },  # best
        ]
        input_constraint, NUM_MAX_ELEMENSTS = None, 9
    elif experiment_type == "ElementSubstitution":
        json_path = "reference_sample_configs/experiment_element_substitution_dict.json"
        param_dict_list = [
            {
                "lr": 0.03,
                "mutate_prob": -1,
                "perturb_prob": -1,
                "perturb_val": -1,
                "max_num_atoms": 5,
            },  # perturb_prob利用
        ]
        input_constraint, NUM_MAX_ELEMENSTS = None, 9
    elif experiment_type == "HSC":
        json_path = (
            "reference_sample_configs/experiment_usual_hsc.json"
        )
        param_dict_list = [
            {
                "lr": 0.04,
                "mutate_prob": 0.35,
                "perturb_prob": 0.03,
                "perturb_val": 0.49,
                "max_num_atoms": 15,
            },  # best
        ]
        input_constraint = np.zeros((1, 118))
        input_constraint[0, 0] = 0.4
        input_constraint = input_constraint.tolist()
        NUM_MAX_ELEMENSTS = 3
    else:
        raise Exception("Not implemented yet")

    with open(json_path, "r") as f:
        experiment_dict_list = json.load(f)

    if test_mode:
        experiment_dict_list = [experiment_dict_list[0]]
        params_dict_list = [param_dict_list[0]]

    all_summary_list = []
    for i, exp_dict in enumerate(experiment_dict_list):
        for j, param_dict in enumerate(param_dict_list):
            print(exp_dict)
            print(param_dict)

            print(f"Experiment {i + 1}/{len(experiment_dict_list)}")
            summary_dict = main_experiment(
                experiment_type=experiment_type,
                exp_dict=exp_dict,
                NUM_MAX_ELEMENSTS=NUM_MAX_ELEMENSTS,
                trial=0,
                mix_alpha=0,
                mutate_prob=param_dict["mutate_prob"],
                purterb_prob=param_dict["perturb_prob"],
                purterb_val=param_dict["perturb_val"],
                max_num_atoms=param_dict["max_num_atoms"],
                lr=param_dict["lr"],
                formation_energy_loss_coef=4.0,
                optimizer="Adam",
                optimization_steps=1000,
                input_constraint=input_constraint,
                test_mode=test_mode,
            )
            all_summary_list.append(summary_dict)

            pd.DataFrame(all_summary_list).to_csv(
                "inverse_summary_result.csv", index=False
            )
