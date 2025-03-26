import ast
import csv
import functools
import itertools
import math
import re
from fractions import Fraction
from functools import reduce
from math import gcd

import numpy as np
import pandas as pd
import smact
import swifter
import torch
from pymatgen.core.composition import Composition
from smact.screening import pauling_test
from tqdm.notebook import tqdm

from .utils import (
    create_atomic_vectors_from_formula_dict,
    define_atom_list,
    elem_list_to_mask,
)


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
    if type(atomic_string) is str:
        return atomic_string.replace(" ", "").replace("_{", "").replace("}", "")
    elif type(atomic_string) is float:
        return "HHeLiBeBCNOFNeNaMgAlSiPS"  # dummy


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


def count_elements(formula_string):
    return len(formula_string.strip().split(" "))


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


def get_evaluate_data_of_element_substitution(npz_path, original_df, NUM_MAX_ELEMENSTS):
    """
    Element Substitutionの評価データを取得する
    """
    # ファイル名から、維持する組成と置換する元素とその酸化数を取得 (English) Get the composition to be maintained and the element to be substituted and its oxidation number from the file name
    (
        remain_atom_dict,
        substitute_target,
        val_of_atom,
        remain_atom_vector,
        substitution_ox_state,
    ) = get_settings_of_element_substitution_in_proposed_method(npz_path)
    # 置換前の組成ベクトルを取得する。 (English) Get the composition vector before substitution.
    original_atom_dict = remain_atom_dict.copy()
    original_atom_dict[substitute_target] = val_of_atom
    original_atom_vector = torch.tensor(
        np.array(create_atomic_vectors_from_formula_dict(original_atom_dict, 0.0)),
        dtype=torch.float32,
    ).to("cuda")
    # Referenceの組成式のSMACT評価 (English) SMACT evaluation of the reference composition formula
    _, reference_formulra_validation, _ = ElectroNegativity_Check(
        NUM_MAX_ELEMENSTS
    ).check(atom_dict_to_smact_string(original_atom_dict))

    # 時間がかかるので、計算対象をスクリーニング (English) Since it takes time, screen the calculation target
    ## 置換の元素数や維持組成の原子数の合計が整数にならないものは除外する。 (English) Exclude those that do not become integers when the number of elements to be substituted and the total number of atoms in the maintained composition are added.
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
    params_dict,
    elem_sub_esp=1e-5,
):
    """
    SMACTによるスクリーニングや、各評価指標の計算を行う
    """

    assert experiment_type in ["ElementSubstitution", "Normal", "HSC"]
    assert method in ["proposed", "SuperDiff"]

    npz_path = csv_path.replace(".csv", ".npz")
    d = np.load(npz_path)

    # formation energy の損失coefを取得
    if method == "proposed" and experiment_type == "ElementSubstitution":
        original_df, original_atom_vector, original_atom_dict, other_data_dict = (
            get_evaluate_data_of_element_substitution(
                npz_path, pd.read_csv(csv_path), NUM_MAX_ELEMENSTS
            )
        )
        ef_coef = float(
            npz_path.split("/")[-2].split("___")[0].split("__results_ef_coef-")[-1]
        )
    elif method == "proposed" and experiment_type == "Normal":
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
    elif method == "proposed" and experiment_type == "HSC":
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

    elif method == "SuperDiff":
        raise Exception("Not implemented yet")
    else:
        raise ValueError(
            f"Invalid method: {method} or experiment_type: {experiment_type}"
        )

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
    n_jobs = min(multiprocessing.cpu_count(), 16)
    results = Parallel(n_jobs=-1)(
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

    # 統計量をdictにまとめる
    sorted_original_atom_dict = {
        k: original_atom_dict[k] for k in sorted(original_atom_dict.keys())
    }
    summary_dict = {
        "base_atom_dict": str(sorted_original_atom_dict),
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

    # すべてのboolsの達成率を計算
    for b_key in bool_keys + ["all_True_bools"]:
        summary_dict[f"{b_key}_rate"] = result_df[b_key].sum() / len(result_df)

    ## Tc の向上幅の平均値を計算
    #### all_True_boolsがTrueのサンプルのみを抽出でTcの差分を辞書データにいれる
    delta_tc = (
        result_df.loc[result_df["all_True_bools"]]["pred_Tc"] - base_pred_Tc
    ).values
    delat_tc = np.sort(delta_tc)[::-1]
    summary_dict["ΔTc_top10"] = np.mean(delta_tc[:10])
    summary_dict["ΔTc_top30"] = np.mean(delta_tc[:30])
    summary_dict["ΔTc_top50"] = np.mean(delta_tc[:50])

    # 実験情報をいれる
    summary_dict["experiment_type"] = experiment_type
    summary_dict["method"] = "proposed"
    summary_dict.update(params_dict)

    result_csv_path = csv_path.replace("optimized_solutions.csv", "result_df.csv")
    result_df.to_csv(result_csv_path, index=False)
    summary_csv_path = csv_path.replace("optimized_solutions.csv", "summary.csv")
    pd.DataFrame([summary_dict]).to_csv(summary_csv_path, index=False)

    return summary_dict


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
    if type(formula) is not str:
        return elements, comps
    for elem_comp in formula.strip().split(" "):
        elem, comp = elem_comp.split("_")
        comp = list(ast.literal_eval(comp))[0]
        elements.append(elem)
        comps.append(round(comp))
    return elements, comps


class check_HSC_heutrality:
    def __init__(self, max_num_atoms, max_num_elements):
        self.max_num_atoms = max_num_atoms
        self.max_num_elements = max_num_elements

    def check_electro_neutrality(
        self, elements, comps, max_num_atoms=20, max_num_elements=3
    ):
        if len(elements) > self.max_num_elements:
            return False, None
        if np.sum(comps) > self.max_num_atoms:
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
    original_df = pd.read_csv(csv_path)
    if method == "proposed":
        original_df["neutral_check_formula"] = original_df[
            "Rounded Optimized Composition"
        ]
    # 一意のもののみを取得する
    unique_df = original_df.drop_duplicates(subset=["neutral_check_formula"]).copy()
    # 電気的中性のチェっっく
    check_neutrality_from_formula_string = check_HSC_heutrality(
        max_num_elements=max_num_elements, max_num_atoms=max_num_atoms
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
    unique_df["num_elements_bools"] = unique_df["neutral_check_formula"].apply(
        lambda x: len(get_list_data_from_formula_string(x)[0])
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
    original_atom_dict = parse_formula(
        csv_path.split("/")[-2]
        .split("___")[0]
        .replace("__results_ef_coef", "")
        .split("-")[0]
    )
    sorted_original_atom_dict = {
        k: original_atom_dict[k] for k in sorted(original_atom_dict.keys())
    }

    # 統計量をdictにまとめる
    summary_dict = {
        "base_atom_dict": str(sorted_original_atom_dict),
        "num_total_samples": len(original_df),
        "unique_rate": len(unique_df) / len(original_df),
        "valid_rate": unique_df["valid_sample_bools"].sum() / len(unique_df),
        "neutral_rate (in valid)": unique_df["neutrality_bools"].sum()
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
    summary_dict["method"] = "proposed"
    summary_dict.update(params_dict)

    result_csv_path = csv_path.replace("optimized_solutions.csv", "result_df.csv")
    result_df.to_csv(result_csv_path, index=False)
    summary_csv_path = csv_path.replace("optimized_solutions.csv", "summary.csv")
    pd.DataFrame([summary_dict]).to_csv(summary_csv_path, index=False)

    return summary_dict
