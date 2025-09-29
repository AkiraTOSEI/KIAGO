#!/usr/bin/env python
# coding: utf-8

# # Convex Hullをformation energy予測器を使って計算するノートブック
# - polar_SMOACSでMatterSim Convex hullで実施したように、MPとAlexでConvex Hullを計算する

# ## ElemNet

# In[1]:


import os

import numpy as np

# In[2]:
import torch
from IPython.display import clear_output, display
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


# In[3]:


elemnet = ElemNet(
    86,
    "1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1",
    activation="relu",
    dropouts=[0.8, 0.9, 0.7, 0.8],
)
elemnet.load_state_dict(torch.load("./models/surrogate_model/elemnet.pth"))
elemnet.eval()
clear_output()
print("pretrained ElemNet is loaded")


# ## 組成式と入力データ操作の関数群

# In[4]:


from typing import List


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


# In[5]:


class Atom_vec2string_converter:
    def __init__(self):
        self.atom_list = define_atom_list()
        self.atomic_number_to_symbol = {
            idx + 1: symbol for idx, symbol in enumerate(self.atom_list)
        }

    def vector2string(self, x: np.ndarray):
        return create_atomic_strings(x, self.atomic_number_to_symbol)


# In[6]:


import ast
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


# ## 凸包計算に必要なデータを取得する関数

# ### Alexandria

# In[8]:


#!/usr/bin/env python3
"""
Alternative implementation using direct HTTP requests for Alexandria OPTIMADE API.

This version uses requests directly for better control over pagination and error handling,
similar to the reference code provided.
"""

import itertools
import logging
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests


def get_formulas_from_chemsys_requests(
    chemsys: List[str], alexandria_base_url: str = "https://alexandria.icams.rub.de/pbe"
) -> List[str]:
    """
    Retrieve all unique chemical formulas for a given chemical system from Alexandria.

    Uses direct HTTP requests for robust pagination handling and error recovery.

    Args:
        chemsys: List of element symbols, e.g., ["Fe", "O"]
        alexandria_base_url: Base URL for Alexandria OPTIMADE endpoint

    Returns:
        List of unique reduced chemical formulas sorted alphabetically

    Raises:
        ConnectionError: If the endpoint is unreachable
        ValueError: If no valid entries are found
    """

    # Step 1: Construct endpoint URL and initialize session
    base_v1 = alexandria_base_url.rstrip("/") + "/v1/"
    structures_url = urljoin(base_v1, "structures")

    session = requests.Session()
    # Configure session with retries
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Test endpoint connectivity
    try:
        test_response = session.get(base_v1, timeout=10)
        test_response.raise_for_status()
        logging.info(f"Successfully connected to Alexandria at {base_v1}")
    except Exception as e:
        raise ConnectionError(f"Cannot reach Alexandria endpoint {base_v1}: {e}")

    formulas = set()

    # Step 2: Generate filters for all subsets of the chemical system
    for subset_size in range(1, len(chemsys) + 1):
        for element_subset in itertools.combinations(chemsys, subset_size):
            # Create OPTIMADE filter for this subset using proper syntax
            # For each subset, we need: nelements=N AND elements HAS "elem1" AND elements HAS "elem2" ...
            element_conditions = [f'elements HAS "{elem}"' for elem in element_subset]
            filter_query = f"nelements={subset_size} AND " + " AND ".join(
                element_conditions
            )

            logging.info(f"Querying subset {element_subset}: {filter_query}")

            try:
                # Step 3: Fetch all pages for this filter
                subset_formulas = _fetch_all_formulas(
                    session, structures_url, filter_query, page_limit=200
                )
                formulas.update(subset_formulas)
                logging.debug(
                    f"Found {len(subset_formulas)} formulas for subset {element_subset}"
                )

            except Exception as e:
                logging.warning(f"Error querying subset {element_subset}: {e}")
                continue  # Continue with other subsets

    # Step 4: Validate and return results
    if not formulas:
        raise ValueError(f"No chemical formulas found for chemical system {chemsys}")

    result = sorted(list(formulas))
    logging.info(f"Total unique formulas found: {len(result)}")
    return result


def _fetch_all_formulas(
    session: requests.Session,
    structures_url: str,
    filter_query: str,
    page_limit: int = 200,
    timeout: Tuple[int, int] = (10, 120),
) -> List[str]:
    """
    Fetch all chemical formulas matching the filter, handling pagination.

    Args:
        session: Configured requests session
        structures_url: OPTIMADE structures endpoint URL
        filter_query: OPTIMADE filter string
        page_limit: Number of entries per page
        timeout: (connect_timeout, read_timeout) in seconds

    Returns:
        List of chemical formulas for this filter
    """
    formulas = []

    # Initial request parameters
    params = {
        "filter": filter_query,
        "response_fields": "chemical_formula_reduced",
        "page_limit": page_limit,
    }

    next_url = structures_url
    next_params = params

    # Handle pagination
    while next_url:
        try:
            response = session.get(next_url, params=next_params, timeout=timeout)
            response.raise_for_status()

            data = response.json()

            # Extract formulas from this page
            for entry in data.get("data", []):
                attributes = entry.get("attributes", {})
                formula = attributes.get("chemical_formula_reduced")
                if formula:
                    formulas.append(formula)

            # Check for next page
            links = data.get("links", {})
            next_url = links.get("next")
            next_params = None  # Next URL is complete, no need for params

            logging.debug(f"Processed page with {len(data.get('data', []))} entries")

        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP request failed: {e}")
            break
        except (KeyError, ValueError) as e:
            logging.error(f"Error parsing response: {e}")
            break

    return formulas


# ### Materials Project

# In[9]:


"""
Script to retrieve all chemical formulas for a given chemical system from Materials Project.
Used for convex hull calculations.

Requirements:
    pip install pymatgen

Main Functions:
    1. get_chemical_formulas_for_chemsys(chemsys) -> List[str]
       - Returns all chemical formulas in a chemical system
       - Example: ["Fe", "O"] → ['Fe', 'FeO', 'Fe2O3', 'Fe3O4', ...]

    2. get_mp_ids_for_formulas(chemsys, target_formulas) -> Dict[str, List[str]]
       - Returns MP IDs for specific formulas
       - Example: ["Fe", "O"], ["Fe3O4"] → {"Fe3O4": ["mp-1271978", ...]}

Example usage:
    formulas = get_chemical_formulas_for_chemsys(["Fe", "O"])
    print(formulas)  # Output: ['Fe', 'FeO', 'Fe2O3', 'Fe3O4', ...]

    mp_ids = get_mp_ids_for_formulas(["Fe", "O"], ["Fe3O4", "Fe17O18"])
    print(mp_ids)    # Output: {"Fe3O4": ["mp-1271978", ...], "Fe17O18": ["mp-705424"]}

Verified to retrieve:
    - Fe17O18 (mp-705424) ✓
    - Fe9O10 (mp-759037) ✓
    - Fe3O4 (mp-1271978) ✓
"""

import logging
from typing import Dict, List, Set, Tuple

from pymatgen.core.composition import Composition
from pymatgen.ext.matproj import MPRester

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Materials Project API key
API_KEY = "e2UzOOWs3I6YIZgPX8VGcfzFkLboyDf1"


def get_chemical_formulas_for_chemsys_MP(
    chemsys: List[str], API_KEY: str, return_mp_ids: bool = False
) -> List[str]:
    """
    Retrieve all chemical formulas for a given chemical system from Materials Project.

    Args:
        chemsys: List of chemical elements (e.g., ["Fe", "O"])
        return_mp_ids: If True, returns tuples of (formula, mp_id) instead of just formulas

    Returns:
        List of reduced chemical formulas as strings (e.g., ["Fe", "O", "FeO", "Fe2O3"])
        Or list of tuples (formula, mp_id) if return_mp_ids=True

    Raises:
        ValueError: If chemsys is empty or contains invalid elements
        ConnectionError: If API request fails
        RuntimeError: If API key is invalid or other API errors occur
    """

    # Input validation
    if not chemsys:
        raise ValueError("Chemical system (chemsys) cannot be empty")

    if not isinstance(chemsys, list):
        raise ValueError("Chemical system must be a list of strings")

    # Convert to set to remove duplicates and sort for consistent querying
    unique_elements = sorted(set(chemsys))
    chemsys_str = "-".join(unique_elements)

    logger.info(f"Querying Materials Project for chemical system: {chemsys_str}")

    try:
        # Initialize MPRester with API key
        with MPRester(API_KEY) as mpr:
            logger.info("Successfully connected to Materials Project API")

            # Query all entries in the chemical system
            # We use the chemsys parameter to get all materials containing only these elements
            entries = mpr.get_entries_in_chemsys(unique_elements)

            if not entries:
                logger.warning(f"No entries found for chemical system: {chemsys_str}")
                return []

            logger.info(
                f"Found {len(entries)} entries for chemical system: {chemsys_str}"
            )

            # Extract unique reduced formulas
            if return_mp_ids:
                # Store formula -> list of mp_ids mapping
                formula_to_ids: Dict[str, List[str]] = {}

                for entry in entries:
                    composition = entry.composition
                    reduced_formula = composition.reduced_formula
                    mp_id = entry.entry_id

                    if reduced_formula not in formula_to_ids:
                        formula_to_ids[reduced_formula] = []
                    formula_to_ids[reduced_formula].append(mp_id)

                # Create list of tuples (formula, representative_mp_id)
                formula_id_list = []
                for formula in sorted(formula_to_ids.keys()):
                    # Use the first (lexicographically smallest) mp_id as representative
                    representative_id = sorted(formula_to_ids[formula])[0]
                    formula_id_list.append((formula, representative_id))

                logger.info(
                    f"Found {len(formula_id_list)} unique chemical formulas with MP IDs"
                )
                return formula_id_list
            else:
                formulas: Set[str] = set()

                for entry in entries:
                    # Get the composition and convert to reduced formula
                    composition = entry.composition
                    reduced_formula = composition.reduced_formula
                    formulas.add(reduced_formula)

                # Convert to sorted list for consistent output
                formula_list = sorted(list(formulas))

                logger.info(f"Found {len(formula_list)} unique chemical formulas")
                logger.info(f"Formulas: {formula_list}")

                return formula_list

    except Exception as e:
        # Handle various types of errors
        error_msg = str(e).lower()

        if "api key" in error_msg or "unauthorized" in error_msg:
            raise RuntimeError(f"Invalid API key or unauthorized access: {e}")
        elif "connection" in error_msg or "network" in error_msg:
            raise ConnectionError(f"Failed to connect to Materials Project API: {e}")
        elif "timeout" in error_msg:
            raise ConnectionError(f"Request timed out: {e}")
        else:
            raise RuntimeError(f"Error querying Materials Project: {e}")


def get_mp_ids_for_formulas(
    chemsys: List[str], target_formulas: List[str], clean_ids: bool = True
) -> Dict[str, List[str]]:
    """
    Get Materials Project IDs for specific chemical formulas in a chemical system.

    Args:
        chemsys: List of chemical elements (e.g., ["Fe", "O"])
        target_formulas: List of chemical formulas to find IDs for (e.g., ["Fe3O4", "Fe17O18"])
        clean_ids: If True, removes calculation method suffixes from MP IDs (default: True)

    Returns:
        Dictionary mapping formula to list of MP IDs (e.g., {"Fe3O4": ["mp-1271978", "mp-12345"]})

    Raises:
        ValueError: If chemsys is empty or contains invalid elements
        ConnectionError: If API request fails
        RuntimeError: If API key is invalid or other API errors occur
    """
    unique_elements = sorted(set(chemsys))
    chemsys_str = "-".join(unique_elements)

    logger.info(
        f"Searching for MP IDs of {target_formulas} in chemical system: {chemsys_str}"
    )

    try:
        with MPRester(API_KEY) as mpr:
            entries = mpr.get_entries_in_chemsys(unique_elements)

            # Build mapping of formula to all its MP IDs
            formula_to_ids: Dict[str, List[str]] = {}

            for entry in entries:
                composition = entry.composition
                reduced_formula = composition.reduced_formula
                mp_id = entry.entry_id

                if reduced_formula not in formula_to_ids:
                    formula_to_ids[reduced_formula] = []
                formula_to_ids[reduced_formula].append(mp_id)

            # Filter for target formulas only
            result = {}
            for formula in target_formulas:
                if formula in formula_to_ids:
                    ids = formula_to_ids[formula]
                    if clean_ids:
                        # Remove calculation method suffixes (e.g., "-GGA+U", "-GGA", etc.)
                        cleaned_ids = []
                        for mp_id in ids:
                            # Keep only the base MP ID (everything before the first dash after "mp-")
                            if mp_id.startswith("mp-"):
                                parts = mp_id.split("-")
                                if len(parts) >= 2:
                                    cleaned_id = f"mp-{parts[1]}"
                                    cleaned_ids.append(cleaned_id)
                                else:
                                    cleaned_ids.append(mp_id)
                            else:
                                cleaned_ids.append(mp_id)
                        # Remove duplicates and sort
                        result[formula] = sorted(list(set(cleaned_ids)))
                    else:
                        result[formula] = sorted(ids)
                else:
                    result[formula] = []

            return result

    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "unauthorized" in error_msg:
            raise RuntimeError(f"Invalid API key or unauthorized access: {e}")
        elif "connection" in error_msg or "network" in error_msg:
            raise ConnectionError(f"Failed to connect to Materials Project API: {e}")
        else:
            raise RuntimeError(f"Error querying Materials Project: {e}")


def validate_elements(elements: List[str]) -> bool:
    """
    Validate that all provided elements are valid chemical symbols.

    Args:
        elements: List of element symbols

    Returns:
        True if all elements are valid, False otherwise
    """
    try:
        for element in elements:
            # Try to create a composition with just this element
            # This will raise an error if the element symbol is invalid
            Composition(element)
        return True
    except Exception:
        return False


# ### 対象材料から凸包計算

# In[10]:


import gzip
import shutil
from typing import Sequence

from pymatgen.core import Element


def chemsys_by_atomic_number(elements: Sequence[str]) -> str:
    """
    Return a hyphen-separated chemical system string with element symbols
    sorted in ascending order of atomic number.

    Parameters
    ----------
    elements : Sequence[str]
        Iterable of element symbols, e.g. ``["Li", "O"]``.

    Returns
    -------
    str
        Hyphen-joined string such as ``"Li-O"``.
    """
    # Sort by atomic number (Element.Z) and join with hyphens
    return "-".join(sorted(elements, key=lambda e: Element(e).Z))


# # 実行ファイルの編集

# In[11]:


from atomate2.vasp.flows.mp import MPGGADoubleRelaxStaticMaker

# In[12]:
from pymatgen.core import Lattice, Structure


def random_structure_from_form_dict(
    form_dict: dict, a: float = 5.0, random_seed: int = None
) -> Structure:
    """
    form_dict (例: {"Fe": 2, "O": 1}) からランダムな構造を生成

    Args:
        form_dict (dict): 元素とその個数
        a (float): 格子定数（立方格子を仮定）
        random_seed (int, optional): 乱数シード（再現性のため）

    Returns:
        Structure: pymatgenのStructureオブジェクト
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 単純に立方格子を仮定
    lattice = Lattice.cubic(a)

    species = []
    coords = []

    for elem, count in form_dict.items():
        for _ in range(int(count)):
            # ランダムな分率座標を生成
            coords.append(np.random.rand(3))
            species.append(elem)

    structure = Structure(lattice, species, coords)
    return structure


# In[51]:


import json

# In[14]:
import re
from pathlib import Path
from typing import Any, Dict

from pymatgen.analysis.phase_diagram import PDPlotter, PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.vasp.inputs import Incar, Potcar, VaspInput

# In[52]:


def calculate_convex_hull_with_elemnet(
    target_formula: str, elemnet, API_KEY, recollect: bool = False
):
    """ """
    # target_formula から原子ベクトルと化学系を決定
    atomic_vec = create_atomic_vectors_from_formula_dict(
        parse_formula(target_formula), val_of_atom=0.0
    )
    print("sum:", np.sum(atomic_vec))
    display(Atom_vec2string_converter().vector2string(np.array([atomic_vec])))
    CHEMSYS = list(np.array(define_atom_list())[np.array(atomic_vec) > 0])

    assert recollect in [False, True], "recollect must be False or True"

    relax_result_db_path = Path(
        "./data/intermidiate/convex_hull/relax_results.json"
    )
    os.makedirs(str(relax_result_db_path.parent), exist_ok=True)
    saved_path = relax_result_db_path.parent / Path(
        f"relax_results_{chemsys_by_atomic_number(CHEMSYS)}.json.gz"
    )
    # Get data from Materials Project and Alexandria Materials Database
    if Path(saved_path).exists() and not recollect:
        with gzip.open(saved_path, "rt", encoding="utf-8") as f:
            all_fetched_data = json.load(f)
        print(
            f"Loaded existing data from {saved_path}, total {len(all_fetched_data)} formulas."
        )
    else:
        ### get data from Materials Project and Alexandria Materials Database
        mp_data = get_chemical_formulas_for_chemsys_MP(
            CHEMSYS, API_KEY
        )  # Materials Project
        alex_data = get_formulas_from_chemsys_requests(
            CHEMSYS
        )  # Alexandria Materials Database
        all_fetched_data = list(set(alex_data) | set(mp_data))

        # all_fetched_dataをjson.gzで保存
        with gzip.open(saved_path, "wt", encoding="utf-8") as f:
            json.dump(all_fetched_data, f, indent=2, ensure_ascii=False)
    # Add target
    all_fetched_data.append(target_formula)
    # ElemNetによるformation Energyの計算
    ### Convert formulas to atomic vectors
    formula_vectors = [
        create_atomic_vectors_from_formula_dict(parse_formula(data), val_of_atom=0.0)
        for data in all_fetched_data
    ]
    formula_vectors = torch.tensor(np.array(formula_vectors, dtype=np.float32))
    ### ElemNetの適用範囲外の元素が入っていないか確認
    assert (formula_vectors[:, 86:] == 0).all()
    formula_vectors = formula_vectors[:, :86]
    ### ElemNetによるformation Energyの計算
    eform = elemnet(torch.tensor(formula_vectors)).to("cpu")
    eform = eform.detach().numpy().flatten()
    ### 単体元素のformation Energyを0にする補正をかける
    simple_idx = (formula_vectors > 0).sum(dim=1) == 1
    display(eform[simple_idx])
    eform[simple_idx] = 0.0
    # Convex Hullの計算
    ### 全ての化学式について、ランダム構造を生成し、ComputedStructureEntryを作成
    entries = []
    for e_i, (chem_f, ef) in enumerate(zip(all_fetched_data, eform)):
        num_atoms = sum(parse_formula(chem_f).values())  # 原子数
        entry = ComputedEntry(
            composition=Composition(chem_f),
            energy=ef * num_atoms,  # energy_pa,      # ← per atom
            entry_id=chem_f,  # ID は一意なら何でも
        )

        if chem_f == target_formula:
            entry_id = e_i
            print(f"{target_formula} formation energy (eV/atom):", ef)
        entries.append(entry)
    phasediag = PhaseDiagram(entries)
    convex_hull_energy = phasediag.get_e_above_hull(entries[entry_id])
    display(convex_hull_energy)
    display(entries[entry_id])

    return convex_hull_energy, phasediag


# ## サンプル

# In[53]:


# # 論文中に提示した材料の評価

# In[55]:


import pandas as pd

# In[56]:


proposed_materials = [
    "O5.0Ca2.0Ba2.0Au2.0",  # Table 3
    "Ce0.0483Ga0.1729Gd0.2178La0.0929Nd0.2338Sm0.2343Ir1.0Si1.0",  # Table 8
]


# In[62]:

API_KEY = None  # Add your API_KEY for materials project!
if API_KEY is None:
    raise Exception("Add your API_KEY for materials project!")


result_list = []
for material in proposed_materials:
    convex_hull_energy, phasediag = calculate_convex_hull_with_elemnet(
        target_formula=material,
        elemnet=elemnet,
        API_KEY=API_KEY,
    )
    result_list.append((material, convex_hull_energy))
    clear_output()
    print(f"{material}: convex hull energy (eV/atom): {convex_hull_energy:.4f}")
df = pd.DataFrame(result_list, columns=["formula", "convex_hull_energy (eV/atom)"])
df.to_csv("proposed_materials_convex_hull.csv", index=False)
display(df)
