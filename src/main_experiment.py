import ast
import os
import shutil
from typing import Optional

from src.config import default_config

from .eval import smact_etc_analysis, smact_etc_analysis_for_HSC
from .inv_opt import InverseOpt4PeriodicTable
from .utils import (
    cfg_add_infos,
    create_atomic_vectors_from_formula_dict,
    create_oxidation_elem_dict,
    define_atom_list,
    define_experiment_chemical_formula,
    elem_list_to_mask,
)


def main_experiment(
    exp_dict: dict,
    NUM_MAX_ELEMENSTS: int,
    experiment_type: str,
    trial: int = 0,
    mix_alpha: float = -1,
    mutate_prob: float = 0.1,
    purterb_prob: float = 0.1,
    purterb_val: float = 0.1,
    formation_energy_loss_coef: Optional[float] = None,
    optimizer: str = "Adam",
    optimization_steps: int = 1000,
    lr: float = 1e-2,
    num_cycle: int = 4,
    input_constraint=None,
    specified_init_candidates=None,
    str_loss_srt: int = 2,
    max_num_atoms: int = 200,
    test_mode: bool = False,
):
    assert experiment_type in ["Normal", "ElementSubstitution", "HSC"]

    cfg = default_config()
    cfg_add_infos(cfg, test_mode=test_mode)
    master_result_dir = f"results4invers/{experiment_type}"
    os.makedirs(master_result_dir, exist_ok=True)

    # 実験設定の読み込み
    if experiment_type == "Normal":
        cform = define_experiment_chemical_formula(exp_dict)
        experiment_name = f"{cform}__results"
    elif experiment_type == "ElementSubstitution":
        sustitute_target = exp_dict["sustitute_target"]
        substitution_ox_state = exp_dict["substitution_ox_state"]
        val_of_atom = exp_dict["val_of_atom"]
        remain_atom_dict = ast.literal_eval(exp_dict["remain_atom_dict"])
        cform = define_experiment_chemical_formula(
            {
                "elems": str(list(remain_atom_dict.keys())),
                "comp": str(list(remain_atom_dict.values())),
            }
        )
        experiment_name = f"ox-{substitution_ox_state}__{cform}__target-{sustitute_target}-{val_of_atom}__results"
    elif experiment_type == "HSC":
        cform = define_experiment_chemical_formula(exp_dict)
        experiment_name = f"{cform}__results"
    else:
        raise Exception("Not implemented yet")

    # 設定の読み込み
    cfg.inverse_problem.initialization.num_mix = -1
    cfg.inverse_problem.initialization.mutate_prob = mutate_prob
    cfg.inverse_problem.initialization.purterb_prob = purterb_prob
    cfg.inverse_problem.initialization.mix_alpha = mix_alpha
    cfg.inverse_problem.initialization.purterb_val = purterb_val

    # loss
    cfg.inverse_problem.loss.structure_loss_srt_epoch = (
        optimization_steps // str_loss_srt
    )
    if formation_energy_loss_coef is not None:
        cfg.inverse_problem.loss.formation_energy_loss_coef = formation_energy_loss_coef
        experiment_name += f"_ef_coef-{formation_energy_loss_coef}"
    else:
        cfg.inverse_problem.loss.formation_energy_loss_coef = formation_energy_loss_coef

    # ハイパラ設定をexperiment_nameに追加
    hyparam_name = f"___{optimizer}-lr-{lr}_optstep-{optimization_steps}_mp-{mutate_prob}_pp-{purterb_prob}_pv-{purterb_val}_mna-{max_num_atoms}_nme-{NUM_MAX_ELEMENSTS}"
    experiment_name += hyparam_name

    # ディレクトリを作る
    result_dir = os.path.join(master_result_dir, experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.output_dirs.inverse_result_dir = result_dir

    # 最大元素数の制約
    cfg.inverse_problem.constraint.num_max_types_of_atoms = NUM_MAX_ELEMENSTS - 1
    cfg.inverse_problem.constraint.num_max_types_of_atoms_constraint_apply_step = (
        optimization_steps // 2
    )

    # Optimizer and steps
    cfg.inverse_problem.method_parameters.optimizer = optimizer
    cfg.inverse_problem.method_parameters.iv_lr = lr
    cfg.inverse_problem.method_parameters.optimization_steps = optimization_steps

    # 解候補の数
    cfg.inverse_problem.method_parameters.iv_batch_size = 2048 * 2

    if experiment_type == "Normal":
        # 入力された組成情報から元素ベクトルを作成する
        atom_dict = dict(
            zip(ast.literal_eval(exp_dict["elems"]), ast.literal_eval(exp_dict["comp"]))
        )
        reference_vector = str(create_atomic_vectors_from_formula_dict(atom_dict, 0.0))
        cfg.inverse_problem.initialization.initializer = reference_vector
        cfg.inverse_problem.loss.structure_list = list(range(1, max_num_atoms))

    elif experiment_type == "ElementSubstitution":
        # get the element list of the substitution oxidation
        oxidation_dict = create_oxidation_elem_dict()
        atomic_opt_mask = elem_list_to_mask(oxidation_dict[substitution_ox_state])
        ### Remove remaining atoms
        all_atom_list = define_atom_list()
        for elem in remain_atom_dict.keys():
            a_idx = all_atom_list.index(elem)
            atomic_opt_mask[a_idx] = 0.0
        ### 設定から、残存させる原子のべくとる(remain_vector)と、酸化数ごとの原子リストの辞書(oxidation_dict)を作成する
        remain_vector = create_atomic_vectors_from_formula_dict(
            remain_atom_dict, val_of_atom
        )
        oxidation_dict = create_oxidation_elem_dict()

        cfg.inverse_problem.initialization.initializer = "random"
        cfg.inverse_problem.constraint.optimization_mask = [
            float(val) for val in atomic_opt_mask
        ]
        cfg.inverse_problem.constraint.input_constraint = [
            float(val) for val in remain_vector
        ]
        # cfg.inverse_problem.loss.structure_list = list(np.arange(1,3)) # structure loss は使わないのでダミー
        cfg.inverse_problem.loss.structure_loss_srt_epoch = (
            1e8  # structure loss は使わない
        )

    elif experiment_type == "HSC":
        assert input_constraint is not None

        cfg.inverse_problem.constraint.input_constraint = input_constraint

        if specified_init_candidates is not None:
            print("use specified_init_candidates in HSC experiment")
            cfg.inverse_problem.initialization.initializer = "specified"
            cfg.inverse_problem.initialization.specified_init_candidates = (
                specified_init_candidates
            )
            cfg.inverse_problem.method_parameters.iv_batch_size = len(
                specified_init_candidates
            )
            cfg.inverse_problem.candidate.num_candidate = len(specified_init_candidates)
        else:
            print("use reference-based initialization in HSC experiment")
            atom_dict = dict(
                zip(
                    ast.literal_eval(exp_dict["elems"]),
                    ast.literal_eval(exp_dict["comp"]),
                )
            )
            reference_vector = str(
                create_atomic_vectors_from_formula_dict(atom_dict, 0.0)
            )
            cfg.inverse_problem.initialization.initializer = reference_vector

        cfg.inverse_problem.display_results = True  # latex出力用
        cfg.inverse_problem.loss.structure_list = list(range(1, max_num_atoms))

    else:
        raise Exception("Not implemented yet")

    # 逆問題最適化の実行
    iv_model, top_N_Tc_mean = InverseOpt4PeriodicTable(cfg, test_mode=test_mode)
    # SMACTによるスクリーニングや、各評価指標の計算を行う
    csv_path = os.path.join(
        cfg.output_dirs.inverse_result_dir, "optimized_solutions.csv"
    )
    if experiment_type == "Normal" or experiment_type == "ElementSubstitution":
        summary_dict = smact_etc_analysis(
            csv_path=csv_path,
            experiment_type=experiment_type,
            method="proposed",
            NUM_MAX_ELEMENSTS=NUM_MAX_ELEMENSTS,
            Ef_criterion=0.0,
            elemnet=iv_model.elemnet.to("cuda"),
            sg_model=iv_model.sg_model.to("cuda"),
            params_dict={
                "mutate_prob": mutate_prob,
                "purterb_prob": purterb_prob,
                "purterb_val": purterb_val,
                "optimizer": optimizer,
                "lr": lr,
            },
            elem_sub_esp=1e-5,
        )
    elif experiment_type == "HSC":
        summary_dict = smact_etc_analysis_for_HSC(
            csv_path=csv_path,
            experiment_type=experiment_type,
            method="proposed",
            Ef_criterion=0.0,
            elemnet=iv_model.elemnet.to("cuda"),
            sg_model=iv_model.sg_model.to("cuda"),
            params_dict={
                "mutate_prob": mutate_prob,
                "purterb_prob": purterb_prob,
                "purterb_val": purterb_val,
                "optimizer": optimizer,
                "lr": lr,
            },
            hydrogen_thres=input_constraint[0][0],
            max_num_atoms=max_num_atoms,
            max_num_elements=NUM_MAX_ELEMENSTS,
        )
    # ログファイルの削除
    shutil.rmtree("lightning_logs", ignore_errors=True)
    shutil.rmtree("logs", ignore_errors=True)
    return summary_dict
