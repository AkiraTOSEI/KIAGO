from omegaconf import DictConfig, OmegaConf


def default_config() -> DictConfig:
    config = {
        "general": {
            "master_dir": "",
            "processed_data_dir": "./",
            "type_of_atoms": 118,
            "deep_speed": None,  # None or "deepspeed_stage_2"
        },
        "output_dirs": {},
        "experiment_names": {},
        "dataset": {
            "dataset_name": None,
            "divide_method": "random",
            "divide_infos": [0.05, 0.15],
            "element_division": True,
            "train_balancing": True,
        },
        "sg_model": {
            "model_type": "resnet18",
            "atom_map_type": "AtomMap-base.npy",
            "fin_act": "relu",
            "optimizer": "Adam",
            "max_epochs": 250,
            "batch_size": 256,
            "lr": 1e-6,
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
            "display_results": False,
            "initialization": {
                "initializer": "dataset",
                "purterb_val": 0.1,
                "purterb_prob": -1,  # 1./118,
                "mutate_prob": -1,  # 0.05,
                "num_mix": 1,
                "mix_alpha": 0.1,
                "min_tc": 30,
                "fixed_mixed_init_data_mode": True,  # 現在でバック中、あとでTrueにする
                "specified_init_candidates": None,
            },
            "loss": {
                "formation_energy_loss_coef": 1.0,
                "structure_loss_srt_epoch": 1000,
                "structure_loss_final_coef": 100000.0,
                "structure_list": list(range(4, 200)),
                "structure_loss_type": "normal",  # "normal" or "variable_composition" NeuLag導入までの一時的な処置
            },
            "constraint": {
                "input_constraint": None,  # None or np.ndarray shape = (1, num_types_of_atoms)、水素化合物半導体の制約などに使う
                "optimization_mask": None,  # None or np.ndarray shape = (1, num_types_of_atoms)、最適化の際に使用するマスク
                "num_max_types_of_atoms": False,
                "num_max_types_of_atoms_constraint_apply_step": 500,  # main script中で設定する
            },
            "method_parameters": {
                "optimizer": "Adam",
                "optimization_steps": 1000,
                "iv_batch_size": 2048 * 8,
                "iv_lr": 1e-4,
                "softmax_temperature": 1.0,
            },
            "candidate": {
                "num_candidate": 2048 * 8,
                "use_candidate_selection": True,
                "reduction_schedule": [1],
            },
            "target_tc": 300,
        },
    }
    cfg = OmegaConf.create(config)
    return cfg
