#!/usr/bin/env python
# coding: utf-8


import json
import os
import re

import pandas as pd
from IPython import display

# In[2]:


df = pd.read_csv(
    "./data/raw_data/SuperCon/primary.tsv",
    sep="\t",
    header=[2],
)
df.head()


# ## 中間データの作成

# In[5]:


# 原子の取得
import periodictable

VALID_ELEMENTS = []
for Z in range(1, 119):
    element = periodictable.elements[Z]
    VALID_ELEMENTS.append(element.symbol)


# In[9]:


def parse_chemical_formula(formula):
    # Split the formula into individual elements and their counts
    elements_and_counts = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", formula)

    # Separate the elements and counts into two lists
    atoms = []
    numbers = []

    for element, count in elements_and_counts:
        if element not in VALID_ELEMENTS:
            # continue  <- ここがバグの原因であった。正しくない原子が抜かれた組成式が帰ってしまう。
            return [], []

        atoms.append(element)
        numbers.append(float(count) if count else 1.0)

    return atoms, numbers


def extract_year_from_string(input_string):
    for year_regex in [r"\b\((\d{4})\)\b", r"\b(\d{4})\b"]:
        matches = re.findall(year_regex, input_string)
        if matches:
            year = int(matches[0])
            if 1900 <= year <= 2100:
                return year

    # cond-mat/YYMMNNN
    for year_regex in [r"cond-mat/\d{2}", r"Cond-Mat/\d{2}"]:
        matches = re.findall(year_regex, input_string)
        if matches:
            year = int(
                "20" + matches[0].replace("cond-mat/", "").replace("Cond-Mat/", "")
            )
            if 1900 <= year <= 2100:
                return year

    if "ISS-96" in input_string:
        return 1996

    print(input_string)
    return None


def SuperConDataProcess(data_path: str, save_path: str):
    """Processes and extracts data from a DataFrame of superconducting materials.

    Args:
        data_path (str): The path to SuperCon dataset CSV file.
        save_path (str): The path to save the processed data as a CSV file.

    Returns:
        None

    Raises:
        AssertionError: If the number of atoms is not equal to the number of numbers.

    Notes:
        - The function assumes the existence of a DataFrame named 'df' with columns 'element' and 'journal'.
        - The function uses helper functions 'parse_chemical_formula' and 'extract_year_from_string' to extract relevant information.
        - Random element and oxygen ratios in the 'element' value are removed during processing.
        - The processed data is saved as a CSV file at the specified 'save_path'.

    """

    df = pd.read_csv(data_path, sep="\t", header=[2])

    original_idx, ATOMS, NUMBERS, YEAR, TC = [], [], [], [], []
    random_ratios = [
        "-Y",
        "+X",
        "+Y",
        "-X",
        "+Z",
        "-Z",
        "-y",
        "-x",
        "+x",
        "+y",
        "-z",
        "+z",
        "-",
        "+",
    ]
    ox_raitos = ["OX", "Ox", "OY", "Oy", "Oz", "OZ"]
    for idx, elem, journal, tc in zip(
        df.index, df.element.values, df.journal.values, df.tc.values
    ):
        if elem == "La1.85Sr0.15Cu1O4-Y":
            verbose = True
        else:
            verbose = False

        remove_flg = False

        # remove random element ratio
        for _rr in random_ratios:
            if _rr in elem:
                remove_flg = True
                if verbose:
                    print(f"{_rr} in {elem}. {elem} is removed.")
            continue

        # remove random Oxygen ratio
        for _rr in ox_raitos:
            if _rr in elem:
                remove_flg = True
                if verbose:
                    print(f"{_rr} in {elem}. {elem} is removed.")

                continue

        if remove_flg:
            continue

        atoms, numbers = parse_chemical_formula(elem)
        year = extract_year_from_string(journal)
        if year is None or len(atoms) == 0:
            continue

        assert len(atoms) == len(numbers)
        ATOMS.append(atoms)
        NUMBERS.append(numbers)
        YEAR.append(year)
        TC.append(tc)
        original_idx.append(idx)
        if verbose:
            print(f"elem: {elem}")
            print(f"atoms: {atoms}")
            print(f"numbers: {numbers}")
            print(f"year: {year}")
            print(f"tc: {tc}")
            print("\n")

    pd.DataFrame([original_idx, ATOMS, NUMBERS, TC, YEAR]).T.to_csv(
        save_path,
        index=False,
        header=["original_idx", "atoms", "numbers", "TC", "year"],
    )


# In[10]:


data_path = "./data/raw_data/SuperCon/primary.tsv"
save_path = "./data/intermediate/PreProcessed_SuperCon.csv"
SuperConDataProcess(data_path, save_path)
