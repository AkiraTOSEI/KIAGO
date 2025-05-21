#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ast
import os
from ast import literal_eval
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal

# from tqdm.notebook import tqdm
from tqdm import tqdm

# In[2]:


intermid_data_dir = "../data/intermediate"
processed_data_dir = "../data/processed"


# # データの統計量を確認
# ## CODデータセット

# In[3]:


cod_df = pd.read_csv(os.path.join(intermid_data_dir, "PreProcess_COD.csv"))
cod_df.sort_values(["year", "atoms"], inplace=True)
cod_df.drop_duplicates(subset=["atoms"], inplace=True)
cod_df.reset_index(drop=True, inplace=True)


# In[4]:


cod_df.head()


# In[5]:


cod_df.describe()


# In[6]:


plt.hist(cod_df.num_types.values, bins=20)
# plt.show()


# ## SuperCon dataset

# In[7]:


sc_df = pd.read_csv(os.path.join(intermid_data_dir, "PreProcessed_SuperCon.csv"))
sc_df.head()


# In[8]:


def sum_numbers(numbers: str) -> int:
    return sum(eval(numbers))


sum_atoms = sc_df.numbers.apply(sum_numbers)
sc_df["sum_atoms"] = sum_atoms
sc_df = sc_df[
    sc_df.sum_atoms < 1000
]  # 1000 個以上の原子を含むものは記載ミスっぽいので削除した
display(sum_atoms.describe())

plt.hist(sum_atoms, bins=50)
plt.title("Number of types of atoms at a data in SuperCon")
plt.yscale("log")
plt.ylabel("Number of data")
plt.show()

plt.hist(sc_df.sum_atoms.values, bins=50)
plt.title("Number of types of atoms at a data in SuperCon")
plt.yscale("log")
plt.ylabel("Number of data")
plt.show()


# In[9]:


def get_num_type_atoms(numbers: str) -> int:
    return len(eval(numbers))


num_type_atoms = sc_df.numbers.apply(get_num_type_atoms)
display(num_type_atoms.describe())
a = plt.hist(num_type_atoms, bins=9)
plt.title("Number of types of atoms in SuperCon")
plt.xlabel("Number of types of atoms")
plt.ylabel("Number of data")
plt.xlim(0, 11)
plt.show()
a


# # データの作成
# - SuperConの統計量を踏まえ、最大の原始種は10にし、それ以上はCODから排除する
# - SuperConデータセット中で、同じ配合でTcが異なるものは平均をとる
# - ランダム分割と、2009年分割の２種類を作る
# - sc_dfに含まれるデータをCODデータセットから排除する
# - データ量が違いすぎるので、SuperConデータセットを水増しする

# In[10]:


augment_ratio = cod_df.shape[0] // sc_df.shape[0]
augment_ratio


# In[11]:


# 原子の取得
import periodictable

VALID_ELEMENTS = []
for Z in range(1, 119):
    element = periodictable.elements[Z]
    VALID_ELEMENTS.append(element.symbol)


# In[12]:


from typing import Dict, List, Tuple

import numpy as np


def atom_names_to_numbers(
    atom_names: List[str], stoichiometry: List[int], VALID_ELEMENTS: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function converts atom names and their stoichiometry into numerical representations.

    Args:
        atom_names (List[str]): A list of atomic names represented as strings.
        stoichiometry (List[int]): A list of atomic stoichiometric coefficients represented as integers.
        VALID_ELEMENTS (List[str]): A list of valid atomic elements. Used for indexing atomic names.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays. The first array represents the atomic presence,
        where each position in the array corresponds to an element in 'VALID_ELEMENTS', and a value of 1 signifies
        the presence of that element in 'atom_names'. The second array represents the stoichiometric counts, where
        each position in the array corresponds to an element in 'VALID_ELEMENTS', and the value is the
        stoichiometric count for that element from 'stoichiometry'.

    Raises:
        IndexError: If an element from 'atom_names' is not found in 'VALID_ELEMENTS'.

    Example:
        atom_names = ["H", "O"]
        stoichiometry = [2, 1]
        VALID_ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        atom_names_to_numbers(atom_names, stoichiometry, VALID_ELEMENTS)
        # Output: (array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0]), array([2, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
    """
    atomic_vector = np.zeros(len(VALID_ELEMENTS))
    stoich_vector = np.zeros(len(VALID_ELEMENTS))
    for atom, number in zip(atom_names, stoichiometry):
        assert atom in VALID_ELEMENTS, f"Element {atom} not found in VALID_ELEMENTS."
        atomic_number = VALID_ELEMENTS.index(atom)
        atomic_vector[atomic_number] = 1
        stoich_vector[atomic_number] = number

    if not np.sum(stoich_vector) >= 1.0:
        nan_vec = np.zeros(len(VALID_ELEMENTS)) * np.nan
        return nan_vec, nan_vec
    return atomic_vector, stoich_vector


# # 超伝導データの処理
# - 重複したデータがあるかを確かめるためのset、実際のデータにわける

# In[13]:


import ast


def get_total_number_of_atoms(numbers):
    numbers = ast.literal_eval(numbers)
    return sum(numbers)


# In[14]:


from typing import Dict, List, Tuple

import numpy as np


def atom_names_to_numbers(
    atom_names: List[str], stoichiometry: List[int], VALID_ELEMENTS: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function converts atom names and their stoichiometry into numerical representations.

    Args:
        atom_names (List[str]): A list of atomic names represented as strings.
        stoichiometry (List[int]): A list of atomic stoichiometric coefficients represented as integers.
        VALID_ELEMENTS (List[str]): A list of valid atomic elements. Used for indexing atomic names.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays. The first array represents the atomic presence,
        where each position in the array corresponds to an element in 'VALID_ELEMENTS', and a value of 1 signifies
        the presence of that element in 'atom_names'. The second array represents the stoichiometric counts, where
        each position in the array corresponds to an element in 'VALID_ELEMENTS', and the value is the
        stoichiometric count for that element from 'stoichiometry'.

    Raises:
        IndexError: If an element from 'atom_names' is not found in 'VALID_ELEMENTS'.

    Example:
        atom_names = ["H", "O"]
        stoichiometry = [2, 1]
        VALID_ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        atom_names_to_numbers(atom_names, stoichiometry, VALID_ELEMENTS)
        # Output: (array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0]), array([2, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
    """
    atomic_vector = np.zeros(len(VALID_ELEMENTS))
    stoich_vector = np.zeros(len(VALID_ELEMENTS))
    for atom, number in zip(atom_names, stoichiometry):
        assert atom in VALID_ELEMENTS, f"Element {atom} not found in VALID_ELEMENTS."
        atomic_number = VALID_ELEMENTS.index(atom)
        atomic_vector[atomic_number] = 1
        stoich_vector[atomic_number] = number

    if np.sum(stoich_vector) >= 1.0 or np.abs(np.sum(stoich_vector) - 1.0) <= 0.011:
        pass
    else:
        nan_vec = np.zeros(len(VALID_ELEMENTS)) * np.nan
        return nan_vec, nan_vec

    return atomic_vector, stoich_vector


# In[15]:


def sort_atoms_numbers(atoms, numbers):
    # アフファベット順に原子を揃える
    atoms = np.array(ast.literal_eval(atoms))
    numbers = np.array(ast.literal_eval(numbers))

    # 重複がある場合はnumbersそのを足す
    if len(atoms) != len(set(atoms)):
        for atom in set(atoms):
            if list(atoms).count(atom) > 1:
                numbers[atoms == atom] = sum(numbers[atoms == atom])
        numbers = np.array([numbers[list(atoms).index(atom)] for atom in set(atoms)])
        atoms = np.array(list(set(atoms)))

    # 0.0がある場合はnanを返す
    if 0.0 in numbers:
        return np.nan, np.nan

    sorted_index = np.argsort(atoms)
    sorted_atoms = str(list(atoms[sorted_index]))
    sorted_numbers = str(list(numbers[sorted_index]))
    return sorted_atoms, sorted_numbers


# In[16]:


class Create_Atomic_Distoribution_Vectors:
    def __init__(self, VALID_ELEMENTS: List[str]):
        self.VALID_ELEMENTS = VALID_ELEMENTS

    def process_row_array(self, atoms, numbers) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a row from the DataFrame.

        Args:
            row (pd.Series): A single row from the DataFrame.

        Returns:
            atomic_vector (np.ndarray): Array representing the atomic vector of the compound.
            stoich_vector (np.ndarray): Array representing the stoichiometric vector of the compound.
        """
        atoms = ast.literal_eval(atoms)
        numbers = ast.literal_eval(numbers)

        _, stoich_vector = atom_names_to_numbers(atoms, numbers, self.VALID_ELEMENTS)
        sum = np.sum(stoich_vector)
        stoich_vector = stoich_vector / np.sum(stoich_vector)
        return stoich_vector, sum

    def process_row_str(self, atoms, numbers) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a row from the DataFrame.

        Args:
            row (pd.Series): A single row from the DataFrame.

        Returns:
            atomic_vector (np.ndarray): Array representing the atomic vector of the compound.
            stoich_vector (np.ndarray): Array representing the stoichiometric vector of the compound.
        """
        atoms = ast.literal_eval(atoms)
        numbers = ast.literal_eval(numbers)
        _, stoich_vector = atom_names_to_numbers(atoms, numbers, self.VALID_ELEMENTS)
        stoich_vector = stoich_vector / np.sum(stoich_vector)

        # 有効数字5桁で文字列化する
        formatted_strings = [f"{x:.5f}" for x in stoich_vector]
        result_string = str(formatted_strings)

        return result_string


# In[17]:


def create_SuperCon_data(
    intermid_data_dir: str,
    processed_data_dir: str,
    VALID_ELEMENTS: List[str],
    element_division: bool = True,
):
    """
    超伝導体のデータセットを作成する
    element_division: 元素の組み合わせベースで分割を行うか。
    """

    # 超伝導体の中間データを読み込む。
    sc_df = pd.read_csv(os.path.join(intermid_data_dir, "PreProcessed_SuperCon.csv"))

    # 1000 個以上の原子を含むものは記載ミスっぽいので削除した
    sc_df["sum_atoms"] = sum_atoms
    sc_df = sc_df[sc_df.sum_atoms < 1000]

    # 原子をアルファベット順に並べた列を作る
    sc_df[["atoms", "numbers"]] = sc_df.apply(
        lambda row: sort_atoms_numbers(row["atoms"], row["numbers"]),
        axis=1,
        result_type="expand",
    )

    # 不自然なデータを削除する
    sc_df = sc_df[~sc_df[["atoms", "numbers"]].isnull().all(axis=1)].reset_index(
        drop=True
    )

    # 同じ組成式で構成されるものは、TCは平均値をとり、発見年は最も古いものを採用する
    tc = sc_df.groupby(["atoms", "numbers"]).mean()["TC"]
    year = sc_df.groupby(["atoms", "numbers"]).min()["year"]
    sc_df = pd.concat([tc, year], axis=1)

    # 同じ組成で構成されるもの（組成比は同じでなくてよい）にIDを振り分け、オリジナルのデータフレームに結合する
    sc_id_df = (
        sc_df.reset_index()["atoms"]
        .drop_duplicates()
        .reset_index()
        .drop("index", axis=1)
        .reset_index()
        .set_index("atoms")
    )
    sc_df = pd.merge(
        sc_df.reset_index().set_index("atoms"),
        sc_id_df,
        left_index=True,
        right_index=True,
        how="inner",
    )
    sc_df.reset_index(inplace=True)
    sc_df.rename(
        {"index": "elem_idx", "TC": "TC(tmp)", "year": "year(tmp)"},
        axis=1,
        inplace=True,
    )

    # 原子分布を得る
    sc_df["dist_string"] = sc_df.apply(
        lambda row: Create_Atomic_Distoribution_Vectors(VALID_ELEMENTS).process_row_str(
            row["atoms"], row["numbers"]
        ),
        axis=1,
        result_type="expand",
    )
    sc_df[["stoich_vector", "stoich_sum"]] = sc_df.apply(
        lambda row: Create_Atomic_Distoribution_Vectors(
            VALID_ELEMENTS
        ).process_row_array(row["atoms"], row["numbers"]),
        axis=1,
        result_type="expand",
    )
    sc_df["original_data_idx"] = np.arange(sc_df.shape[0])
    sc_vector_df = sc_df[["original_data_idx", "stoich_vector"]]
    sc_vector_df.set_index("original_data_idx", inplace=True)
    sc_df.drop("stoich_vector", axis=1, inplace=True)

    # H2C2, H4C4など比率が同じでも組成の数が異なるものは、TC、YEARの処理がでてきないので、原子分布の文字列(dist_string)をもとに処理し直す
    # 同じ組成式で構成されるものは、TCは平均値をとり、発見年は最も古いものを採用する
    tc = sc_df.groupby("dist_string").mean()["TC(tmp)"]
    year = sc_df.groupby("dist_string").min()["year(tmp)"]
    tc.name = "Tc"
    year.name = "year"
    # 処理したTC、YEARをつなげる
    sc_df = pd.merge(
        pd.concat([tc, year], axis=1),
        sc_df.set_index("dist_string"),
        left_index=True,
        right_index=True,
        how="inner",
    )

    # データの重複, 原子分布のnanを削除する
    non_duplicated_idx = ~sc_df.index.duplicated(keep="first")
    non_na_idx = sc_df.index != str(["nan"] * 118)
    _bool = non_duplicated_idx * non_na_idx
    sc_df = sc_df[_bool]
    sc_df_array = sc_vector_df.loc[sc_df["original_data_idx"].values]

    # 整理したデータフレームと、配列の原子分布の突き合わせを行う。
    # データの順序があっているかをテストする
    num_test = 200
    dist_string_array = np.array(sc_df.index)
    sc_df_array = np.stack(sc_df_array.stoich_vector.values)
    for data_i in np.random.choice(sc_df_array.shape[0], num_test, replace=False):
        np.testing.assert_almost_equal(
            np.array(ast.literal_eval(dist_string_array[data_i])).astype(float),
            sc_df_array[data_i],
            decimal=5,
        )

    # indexを振り直す
    sc_df.elem_idx = sc_df.elem_idx + 1
    sc_df["data_id"] = np.arange(sc_df.shape[0]) + 1

    if element_division:
        saved_fname = "SuperCon_data.npz"
    else:
        # 元素の組み合わせでデータ分割しないときは、elem_idxをdata_idと一致させる
        saved_fname = "SuperCon_data_NoElementDivision.npz"
        sc_df.elem_idx = sc_df.data_id

    # データを保存する
    np.savez(
        os.path.join(processed_data_dir, saved_fname),
        sc_X=sc_df_array,
        sc_Y=sc_df.Tc.values,
        sc_YEARS=sc_df.year.values,
        data_ids=sc_df.data_id.values,
        elem_ids=sc_df.elem_idx.values,
        total_num=sc_df["numbers"].apply(get_total_number_of_atoms).values,
    )

    return sc_df


# # CODデータの処理

# In[18]:


def create_COD_data(
    intermid_data_dir: str,
    processed_data_dir: str,
    VALID_ELEMENTS: List[str],
    sc_df: pd.DataFrame,
    element_division: bool = True,
):
    """
    非超伝導体のデータセットを作成する
    """
    # CODの中間処理データを読みこむ
    cod_df = pd.read_csv(os.path.join(intermid_data_dir, "PreProcess_COD.csv"))
    cod_df["TC"] = 0.0
    # 原子をアルファベット順に並べた列を作る
    cod_df[["atoms", "numbers"]] = cod_df.apply(
        lambda row: sort_atoms_numbers(row["atoms"], row["numbers"]),
        axis=1,
        result_type="expand",
    )

    # 不自然なデータを削除する
    cod_df = cod_df[~cod_df[["atoms", "numbers"]].isnull().all(axis=1)].reset_index(
        drop=True
    )

    # sc_df に存在しない cod_df のインデックスを取得することで、超伝導データをcod_dfから排除する
    cod_df = cod_df.set_index("atoms")
    missing_indexes = cod_df.index.difference(sc_df.set_index("atoms").index)
    cod_df = cod_df.loc[missing_indexes].reset_index()

    # 同じ組成式で構成されるものは、発見年は最も古いものを採用する
    tc = cod_df.groupby(["atoms", "numbers"]).mean()[
        "TC"
    ]  # Tcはゼロだが、SuperConと同じコードを使うため、こうしている
    year = cod_df.groupby(["atoms", "numbers"]).min()["year"]
    cod_df = pd.concat([tc, year], axis=1)

    # 同じ組成で構成されるもの（組成比は同じでなくてよい）にIDを振り分け、オリジナルのデータフレームに結合する
    cod_id_df = (
        cod_df.reset_index()["atoms"]
        .drop_duplicates()
        .reset_index()
        .drop("index", axis=1)
        .reset_index()
        .set_index("atoms")
    )
    cod_df = pd.merge(
        cod_df.reset_index().set_index("atoms"),
        cod_id_df,
        left_index=True,
        right_index=True,
        how="inner",
    )
    cod_df.reset_index(inplace=True)
    cod_df.rename(
        {"index": "elem_idx", "TC": "TC(tmp)", "year": "year(tmp)"},
        axis=1,
        inplace=True,
    )
    # 原子分布を得る
    cod_df["dist_string"] = cod_df.apply(
        lambda row: Create_Atomic_Distoribution_Vectors(VALID_ELEMENTS).process_row_str(
            row["atoms"], row["numbers"]
        ),
        axis=1,
        result_type="expand",
    )
    cod_df[["stoich_vector", "stoich_sum"]] = cod_df.apply(
        lambda row: Create_Atomic_Distoribution_Vectors(
            VALID_ELEMENTS
        ).process_row_array(row["atoms"], row["numbers"]),
        axis=1,
        result_type="expand",
    )
    cod_df["original_data_idx"] = np.arange(cod_df.shape[0])

    # vecotrを分離する
    cod_vector_df = cod_df[["original_data_idx", "stoich_vector"]]
    cod_vector_df.set_index("original_data_idx", inplace=True)
    cod_df.drop("stoich_vector", axis=1, inplace=True)

    # H2C2, H4C4など比率が同じでも組成の数が異なるものは、TC、YEARの処理がでてきないので、原子分布の文字列(dist_string)をもとに処理し直す
    # 同じ組成式で構成されるものは、TCは平均値をとり、発見年は最も古いものを採用する
    tc = cod_df.groupby("dist_string").mean()["TC(tmp)"]
    year = cod_df.groupby("dist_string").min()["year(tmp)"]
    tc.name = "Tc"
    year.name = "year"
    # 処理したTC、YEARをつなげる
    cod_df = pd.merge(
        pd.concat([tc, year], axis=1),
        cod_df.set_index("dist_string"),
        left_index=True,
        right_index=True,
        how="inner",
    )

    # データの重複, 原子分布のnanを削除する
    non_duplicated_idx = ~cod_df.index.duplicated(keep="first")
    non_na_idx = cod_df.index != str(["nan"] * 118)
    _bool = non_duplicated_idx * non_na_idx
    cod_df = cod_df[_bool]
    cod_df_array = cod_vector_df.loc[cod_df["original_data_idx"].values]

    # 整理したデータフレームと、配列の原子分布の突き合わせを行う。
    # データの順序があっているかをテストする
    num_test = 200
    dist_string_array = np.array(cod_df.index)
    cod_df_array = np.stack(cod_df_array.stoich_vector.values)
    for data_i in np.random.choice(cod_df_array.shape[0], num_test, replace=False):
        np.testing.assert_almost_equal(
            np.array(ast.literal_eval(dist_string_array[data_i])).astype(float),
            cod_df_array[data_i],
            decimal=5,
        )

    # indexを振り直す
    cod_df.elem_idx = -1 * (cod_df.elem_idx + 1)
    cod_df["data_id"] = -1 * (np.arange(cod_df.shape[0]) + 1)

    if element_division:
        saved_fname = "COD_data.npz"
    else:
        # 元素の組み合わせでデータ分割しないときは、elem_idxをdata_idと一致させる
        saved_fname = "COD_data_NoElementDivision.npz"
        cod_df.elem_idx = cod_df.data_id

    # データを保存する
    np.savez(
        os.path.join(processed_data_dir, saved_fname),
        cod_X=cod_df_array,
        cod_Y=cod_df.Tc.values,
        cod_YEARS=cod_df.year.values,
        data_ids=cod_df.data_id.values,
        elem_ids=cod_df.elem_idx.values,
        total_num=cod_df["numbers"].apply(get_total_number_of_atoms).values,
    )
    return cod_df


# # SuperCon, CODをまとめてデータセットにする

# In[19]:


from typing import List


def time_data_split(
    YEARS: np.ndarray, first_valid_year: int, first_test_year: int
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Split data based on time years.

    Args:
        YEARS (numpy.ndarray): An array of years corresponding to each data point.
        first_valid_year (int): The first year to be used for validation.
        first_test_year (int): The first year to be used for testing.

    Returns:
        tuple of numpy.ndarray: Boolean arrays for training, validation, and testing sets.
    """
    assert type(first_test_year) == int and type(first_valid_year) == int
    assert first_valid_year < first_test_year

    train_bool = np.array(YEARS < first_valid_year)
    valid_bool = np.array((first_valid_year <= YEARS) & (YEARS < first_test_year))
    test_bool = np.array(first_test_year <= YEARS)

    return train_bool, valid_bool, test_bool


def random_data_split(
    data_ids: np.ndarray, valid_ratio: float = 0.037, test_ratio: float = 0.193
):
    """
    Divide a dataset into train, validation, and test subsets.

    Args:
        dataset (numpy.ndarray): The dataset to be divided.
        data_ids(numpy.ndarray): The data ids corresponding to each data point.
        valid_ratio (float): The ratio of the dataset to be used for validation.
        test_ratio (float): The ratio of the dataset to be used for testing.

    Returns:
        tuple of numpy.ndarray: Boolean arrays for training, validation, and testing sets.
    """

    # Ensure the sum of ratios is less than 1
    if valid_ratio + test_ratio >= 1:
        raise ValueError("The sum of validation and test ratios should be less than 1")

    # Shuffle the dataset indices
    unique_ids = np.unique(data_ids)
    np.random.shuffle(unique_ids)
    unique_id_size = len(unique_ids)

    valid_size = int(valid_ratio * unique_id_size)
    test_size = int(test_ratio * unique_id_size)

    valid_ids = unique_ids[:valid_size]
    test_ids = unique_ids[valid_size : valid_size + test_size]
    train_ids = unique_ids[valid_size + test_size :]

    valid_bool = np.isin(data_ids, valid_ids)
    test_bool = np.isin(data_ids, test_ids)
    train_bool = np.isin(data_ids, train_ids)

    # check
    for train_id in data_ids[train_bool]:
        assert train_id not in data_ids[valid_bool]
        assert train_id not in data_ids[test_bool]

    assert valid_bool.sum() + test_bool.sum() + train_bool.sum() == len(data_ids)

    return train_bool, valid_bool, test_bool


# In[20]:


def data_split(
    X: np.ndarray,
    Y: np.ndarray,
    YEARS: np.ndarray,
    data_ids: np.ndarray,
    elem_ids: np.ndarray,
    total_num: np.ndarray,
    divide_infos: List[any],
    datatype: str,
    method: str = "time",
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split data into train, validation and test set.

    Args:
        X (numpy.ndarray): The input data to be divided.
        Y (numpy.ndarray): The target data to be divided.
        divide_infos (List[any]): A list containing information to be used for dividing the data.
        datatype (str): The type of data.
        method (str, optional): The method to be used for splitting the data. Options: 'time', 'random'. Default: 'time'.

    Returns:
        tuple of numpy.ndarray: Arrays for training, validation, and testing sets for both input and target data.
    """

    if method == "time":
        train_bool, valid_bool, test_bool = time_data_split(
            YEARS, divide_infos[0], divide_infos[1]
        )
    elif method == "random":
        train_bool, valid_bool, test_bool = random_data_split(
            elem_ids, divide_infos[0], divide_infos[1]
        )
    else:
        raise Exception("Invalid method")

    print(X.shape, elem_ids.shape)

    print(
        f"{datatype} num train data: {len(X[train_bool])}, ratio:{round(len(X[train_bool]) / len(X), 3)}"
    )
    print(
        f"{datatype} num valid data: {len(X[valid_bool])}, ratio:{round(len(X[valid_bool]) / len(X), 3)}"
    )
    print(
        f"{datatype} num test data: {len(X[test_bool])},ratio:{round(len(X[test_bool]) / len(X), 3)}"
    )
    train_dataset = [
        X[train_bool],
        Y[train_bool],
        YEARS[train_bool],
        data_ids[train_bool],
        elem_ids[train_bool],
        total_num[train_bool],
    ]
    valid_dataset = [
        X[valid_bool],
        Y[valid_bool],
        YEARS[valid_bool],
        data_ids[valid_bool],
        elem_ids[valid_bool],
        total_num[valid_bool],
    ]
    test_dataset = [
        X[test_bool],
        Y[test_bool],
        YEARS[test_bool],
        data_ids[test_bool],
        elem_ids[test_bool],
        total_num[test_bool],
    ]

    return train_dataset, valid_dataset, test_dataset


# In[21]:


from typing import List, Tuple, Union


def load_dataset(
    processed_data_dir: str,
    divide_method: str,
    first_valid_year: int,
    first_test_year: int,
    valid_ratio: float,
    test_ratio: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Union[int, float]]]:
    """
    Loads SuperCon and COD datasets, determines dividing information based on
    the divide method, and splits SuperCon and COD data.

    Args:
        intermid_data_dir (str): The directory where the data files are stored.
        divide_method (str): Method to be used for splitting the data. Options: 'time', 'random'.
        first_test_year (int): The first year to be included in the test set.
        first_valid_year (int): The first year to be included in the validation set.
        valid_ratio (float): The proportion of data to be used for validation in random splitting.
        test_ratio (float): The proportion of data to be used for testing in random splitting.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Returns a tuple where the first element
        is a list of split SuperCon data and the second element is a list of split COD data.

    Raises:
        ValueError: If an invalid divide method is provided, or necessary arguments for the divide method are missing.
    """

    # Validate divide_method input
    valid_methods = ["time", "random"]
    if divide_method not in valid_methods:
        raise ValueError(f"Invalid divide method. Expected one of: {valid_methods}")

    if divide_method == "time" and (
        first_valid_year is None or first_test_year is None
    ):
        raise ValueError(
            "First year for validation and test sets must be provided when divide method is 'time'"
        )

    if divide_method == "random" and (valid_ratio is None or test_ratio is None):
        raise ValueError(
            "Valid and test ratios must be provided when divide method is 'random'"
        )

    # Load SuperCon data
    if element_division:
        cod_fname = "COD_data.npz"
        sc_fname = "SuperCon_data.npz"
    else:
        cod_fname = "COD_data_NoElementDivision.npz"
        sc_fname = "SuperCon_data_NoElementDivision.npz"
    sc_d = np.load(os.path.join(processed_data_dir, sc_fname))
    sc_X = sc_d["sc_X"]
    sc_Y = sc_d["sc_Y"]
    sc_YEARS = sc_d["sc_YEARS"]
    sc_data_ids = sc_d["data_ids"]
    sc_elem_ids = sc_d["elem_ids"]
    sc_total_num = sc_d["total_num"]

    # Load COD data
    cod_d = np.load(os.path.join(processed_data_dir, cod_fname))
    cod_X = cod_d["cod_X"]
    cod_Y = cod_d["cod_Y"]
    cod_YEARS = cod_d["cod_YEARS"]
    cod_data_ids = cod_d["data_ids"]
    cod_elem_ids = cod_d["elem_ids"]
    cod_total_num = cod_d["total_num"]

    # Determine divide information based on the divide method
    divide_infos = (
        [first_valid_year, first_test_year]
        if divide_method == "time"
        else [valid_ratio, test_ratio]
    )

    # Split SuperCon and COD data
    sc_dataset = data_split(
        X=sc_X,
        Y=sc_Y,
        YEARS=sc_YEARS,
        data_ids=sc_data_ids,
        elem_ids=sc_elem_ids,
        total_num=sc_total_num,
        divide_infos=divide_infos,
        datatype="SuperCon",
        method=divide_method,
    )
    cod_dataset = data_split(
        X=cod_X,
        Y=cod_Y,
        YEARS=cod_YEARS,
        data_ids=cod_data_ids,
        elem_ids=cod_elem_ids,
        total_num=cod_total_num,
        divide_infos=divide_infos,
        datatype="COD",
        method=divide_method,
    )

    return sc_dataset, cod_dataset, divide_infos


# In[33]:


import copy
import os
from typing import Dict, List, Tuple, Union

import numpy as np


def prepare_dataset(
    intermid_data_dir: str,
    processed_data_dir: str,
    train_balancing: bool = True,
    divide_method: str = "time",
    element_division: bool = True,
    first_test_year: int = 2010,  #  FeSC発見タスク:2008, 細野先生精度確認タスク: 2010
    first_valid_year: int = 2008,  # FeSC発見タスク:2005, 細野先生精度確認タスク: 2008
    valid_ratio: float = 0.05,  # 0.037,
    test_ratio: float = 0.15,  # 0.193,
) -> None:
    """
    Prepares the dataset by splitting it into training, validation, and testing sets,
    and saves the dataset to disk.

    Args:
        processed_data_dir (str): The directory to save the processed data.
        intermid_data_dir (str): The directory where the intermediate data is stored.
        test_balancing (bool): If true, balance the training data.
        divide_method (str): The method to be used for splitting the data. Options: 'time', 'random'.
        first_test_year (int): The first year to be included in the test set.
        first_valid_year (int): The first year to be included in the validation set.
        valid_ratio (float): The proportion of data to be used for validation in random splitting.
        test_ratio (float): The proportion of data to be used for testing in random splitting.

    Returns:
        None
    """

    # Load and split datasets
    sc_dataset, cod_dataset, divide_infos = load_dataset(
        processed_data_dir,
        divide_method,
        first_valid_year,
        first_test_year,
        valid_ratio,
        test_ratio,
    )

    # Merge SuperCon and COD data
    dataset = dict()
    for phase, sc_data, cod_data in zip(
        ["train", "valid", "test"], sc_dataset, cod_dataset
    ):
        sc_x, sc_y, sc_years, sc_ids, sc_elem_ids, sc_total_num = sc_data
        cod_x, cod_y, cod_years, cod_ids, cod_elem_ids, cod_total_num = cod_data

        assert cod_y.max() == 0.0

        # Training data balancing
        if train_balancing and phase == "train":
            cod_sc_train_ratio = np.floor(cod_x.shape[0] / sc_x.shape[0]).astype(int)
            print(f"cod/sc train ratio: {cod_sc_train_ratio}")
            sc_x = np.repeat(sc_x, cod_sc_train_ratio, axis=0)
            sc_y = np.repeat(sc_y, cod_sc_train_ratio, axis=0)
            sc_years = np.repeat(sc_years, cod_sc_train_ratio, axis=0)
            sc_ids = np.repeat(sc_ids, cod_sc_train_ratio, axis=0)
            sc_elem_ids = np.repeat(sc_elem_ids, cod_sc_train_ratio, axis=0)
            sc_total_num = np.repeat(sc_total_num, cod_sc_train_ratio, axis=0)

        if phase == "train":
            print("traning data shuffling")
            shfl = np.random.choice(np.arange(len(sc_x)), len(sc_x), replace=False)
            sc_x = sc_x[shfl]
            sc_y = sc_y[shfl]
            sc_years = sc_years[shfl]
            sc_ids = sc_ids[shfl]
            sc_total_num = sc_total_num[shfl]

        # Concatenate SuperCon and COD data
        x = np.concatenate([sc_x, cod_x], axis=0)
        y = np.concatenate([sc_y, cod_y], axis=0)
        years = np.concatenate([sc_years, cod_years], axis=0)
        ids = np.concatenate([sc_ids, cod_ids], axis=0)
        elem_ids = np.concatenate([sc_elem_ids, cod_elem_ids], axis=0)
        total_num = np.concatenate([sc_total_num, cod_total_num], axis=0)
        print(f"num {phase} data: {len(x)}")

        data = (x, y, years, ids, elem_ids, total_num)
        data = copy.deepcopy(data)
        dataset[phase] = data

    # check
    assert len(set(dataset["train"][3]).intersection(set(dataset["valid"][3]))) == 0
    assert len(set(dataset["train"][3]).intersection(set(dataset["test"][3]))) == 0
    assert len(set(dataset["train"][4]).intersection(set(dataset["valid"][4]))) == 0
    assert len(set(dataset["train"][4]).intersection(set(dataset["test"][4]))) == 0

    # Save the dataset
    file_name = f"{divide_method}-{divide_infos[0]}-{divide_infos[1]}"
    if not element_division:
        file_name += "-NoElementDivision"

    if train_balancing:
        file_name += "-balancing"

    saved_path = os.path.join(processed_data_dir, f"{file_name}.npz")
    print("saved path:", saved_path)
    np.savez(
        saved_path,
        x_train=dataset["train"][0],
        y_train=dataset["train"][1],
        years_train=dataset["train"][2],
        ids_train=dataset["train"][3],
        elem_ids_train=dataset["train"][4],
        total_num_train=dataset["train"][5],
        x_valid=dataset["valid"][0],
        y_valid=dataset["valid"][1],
        years_valid=dataset["valid"][2],
        ids_valid=dataset["valid"][3],
        elem_ids_valid=dataset["valid"][4],
        total_num_valid=dataset["valid"][5],
        x_test=dataset["test"][0],
        y_test=dataset["test"][1],
        years_test=dataset["test"][2],
        ids_test=dataset["test"][3],
        elem_ids_test=dataset["test"][4],
        total_num_test=dataset["test"][5],
    )


# # データの準備
#
# ## SuperCon, CODデータそれぞれの処理

# In[34]:


element_division = True
TEST_sc_df = create_SuperCon_data(
    intermid_data_dir, processed_data_dir, VALID_ELEMENTS, element_division
)
TEST_cod_df = create_COD_data(
    intermid_data_dir, processed_data_dir, VALID_ELEMENTS, TEST_sc_df, element_division
)


# In[35]:


methods = ["random"]
for divide_method in methods:
    for train_balancing in [True]:
        print("Train balancing:", train_balancing)
        print("Divide method:", divide_method)
        prepare_dataset(
            intermid_data_dir,
            processed_data_dir,
            train_balancing,
            divide_method,
            element_division=element_division,
        )
        print("\n \n \n \n")
