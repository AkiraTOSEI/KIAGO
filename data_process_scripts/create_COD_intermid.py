#!/usr/bin/env python
# coding: utf-8


# In[1]:


import multiprocessing
import os
import random
import re
import warnings
from typing import List, Set, Tuple

import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.io.feff import Header
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")


# In[2]:


VALID_ELEMENTS = (
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
    "K",
    "Ar",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Ni",
    "Co",
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
)


# In[ ]:


# # CIFファイルをパースして構造オブジェクトを取得します
#

# In[3]:


def get_cif_files(directory):
    cif_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cif"):
                cif_files.append(os.path.join(root, file))

    if len(cif_files) == 0:
        raise Exception("No cif files!")
    return cif_files


# In[4]:


def get_year(string: str):
    """
    Extracts the 4-digit year value from a BibTeX string.

    Args:
        string (str): A BibTeX string containing publication information.

    Returns:
        int or None: The 4-digit year value found in the BibTeX string, or None if not found.

    Example:
        string = '@article{cifref0,\n    author = "Taguchi, Haruki",\n    title = "  Concise syntheses of (--)-habiterpenol and (+)-2,3-epi-habiterpenol via  redox radical cyclization of alkenylsilane",\n    journal = "Organic \\&amp; Biomolecular Chemistry",\n    year = "2023"\n}\n'
        year = get_year(string)
        # Output: 2023

    Note:
        The function assumes that the BibTeX string is well-formed and contains a line with the 'year' information.
        If the 4-digit year is not found in the BibTeX string, the function returns None.
    """

    string = string.strip()

    for elem in string.split("\n"):
        elem = elem.strip()
        if "year" in elem:
            key, val = elem.split(" = ")
            year_match = re.search(r"\d{4}", elem)
            if year_match:
                year = year_match.group()
                return int(year)

    print("No 4-digit year found.")
    return None


# In[5]:


def extract_atoms_and_numbers(
    atom_list: List[str], VALID_ELEMENTS: Set[str]
) -> Tuple[List[str], List[int]]:
    """
    Extracts atoms and their corresponding numbers from a list of strings.

    Args:
        atom_list (list): A list of strings representing atoms and their numbers, e.g., ['H68', 'C84', 'S8', 'N4'].
        VALID_ELEMENTS (set): A set of valid elements (atoms), e.g., {'H', 'He', 'Li', ...}.

    Returns:
        tuple: A tuple containing two lists:
            1. List of atoms (str): e.g., ['H', 'C', 'S', 'N']
            2. List of numbers (int): e.g., [68, 84, 8, 4]

    Raises:
        ValueError: If an invalid atom is encountered in the input list.
    """
    atoms: List[str] = []
    numbers: List[int] = []

    for atom_info in atom_list:
        atom = "".join(filter(str.isalpha, atom_info))
        number = "".join(filter(str.isdigit, atom_info))

        # Check if the atom and number are not empty
        if not atom or not number:
            raise ValueError(f"Invalid atom: {atom_info}")

        # Check if the atom is in the VALID_ELEMENTS set
        if atom not in VALID_ELEMENTS:
            raise ValueError(f"Invalid atom: {atom} in {atom_info}")

        atoms.append(atom)
        numbers.append(int(number))

    return atoms, numbers


# In[6]:


def get_formula_and_year(
    cif_file_path: str, VALID_ELEMENTS: Set[str]
) -> Tuple[List[str], List[int], int]:
    """
    Extracts the chemical formula and year information from a CIF file.

    Args:
        cif_file_path (str): The path to the CIF file.
        VALID_ELEMENTS (set): A set of valid elements (atoms) used for extracting the chemical formula.

    Returns:
        tuple: A tuple containing three elements:
            1. List of atoms (str): The chemical elements extracted from the formula.
            2. List of numbers (int): The corresponding numbers of atoms in the formula.
            3. int or None: The 4-digit year value found in the CIF's BibTeX information, or None if not found.

    Example:
        cif_file_path = "/path/to/cif_file.cif"
        VALID_ELEMENTS = {'H', 'C', 'S', 'N', 'O'}  # Set of valid elements for the formula extraction
        atoms, numbers, year = get_formula_and_year(cif_file_path, VALID_ELEMENTS)

    Note:
        The function uses pymatgen library to read the CIF file and extract formula information.
        If the formula cannot be extracted or no valid year information is found, it returns None for both formula and year.
        The function assumes that the CIF file is well-formed and contains valid chemical formula information and BibTeX data.
    """
    try:
        header = Header.from_cif_file(cif_file_path)
        formula = header.formula
        atoms, numbers = extract_atoms_and_numbers(formula.split(" "), VALID_ELEMENTS)
    except:
        return None, None, None

    # get year from bibtex string
    parser = CifParser(cif_file_path)
    year = get_year(parser.get_bibtex_string())

    if year is not None:
        return atoms, numbers, year


# In[ ]:


# In[7]:


def process_cif_files(
    cif_files_list: List[str], VALID_ELEMENTS: Set[str]
) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """
    Processes a list of CIF files to extract chemical formulas and year information.

    Args:
        cif_files_list (list): A list of paths to CIF files.
        VALID_ELEMENTS (set): A set of valid elements (atoms) used for extracting the chemical formula.

    Returns:
        tuple: A tuple containing three lists:
            1. List of lists of atoms (list): A list of lists containing the chemical elements extracted from the formulas of each CIF file.
            2. List of lists of numbers (list): A list of lists containing the corresponding numbers of atoms in the formulas of each CIF file.
            3. List of integers (list): A list containing the 4-digit year values found in the BibTeX information of each CIF file.

    Example:
        cif_files_list = ["/path/to/cif_file1.cif", "/path/to/cif_file2.cif", ...]
        VALID_ELEMENTS = {'H', 'C', 'S', 'N', 'O'}  # Set of valid elements for the formula extraction
        atoms, numbers, years = process_cif_files(cif_files_list, VALID_ELEMENTS)

    Note:
        The function iterates through the list of CIF files, extracting the chemical formula and year information using get_formula_and_year function.
        If an error occurs while processing a CIF file, the function prints an error message and exits the loop.
        The function limits the number of processed CIF files to 10,000 (can be adjusted with the 'i' variable).
    """
    ATOMS: List[List[str]] = []
    NUMBERS: List[List[int]] = []
    YEARS: List[int] = []

    for cif_file_path in tqdm(cif_files_list):
        try:
            atoms, numbers, year = get_formula_and_year(cif_file_path, VALID_ELEMENTS)
            ATOMS.append(atoms)
            NUMBERS.append(numbers)
            YEARS.append(year)
        except Exception as e:
            print(f"Error processing CIF file: {cif_file_path}")
            print(f"Error message: {e}")
            break

    return ATOMS, NUMBERS, YEARS


def process_cif_file(cif_file_path, VALID_ELEMENTS):
    """
    Processes a single CIF file to extract chemical formulas and year information.

    Args:
        cif_file_path (str): The path to the CIF file.
        VALID_ELEMENTS (set): A set of valid elements (atoms) used for extracting the chemical formula.

    Returns:
        tuple: A tuple containing three elements:
            1. List of atoms (list): The chemical elements extracted from the formula.
            2. List of numbers (list): The corresponding numbers of atoms in the formula.
            3. int or None: The 4-digit year value found in the CIF's BibTeX information, or None if not found.

    Note:
        This function calls get_formula_and_year to process a single CIF file and extract chemical formulas and year information.
        If an error occurs while processing the CIF file, the function prints an error message and returns None for the respective data.
    """
    try:
        atoms, numbers, year = get_formula_and_year(cif_file_path, VALID_ELEMENTS)
        return atoms, numbers, year
    except Exception as e:
        print(f"Error processing CIF file: {cif_file_path}")
        print(f"Error message: {e}")
        return None, None, None


def process_cif_files(
    cif_files_list: List[str], VALID_ELEMENTS: Set[str]
) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """
    Processes a list of CIF files to extract chemical formulas and year information.

    Args:
        cif_files_list (list): A list of paths to CIF files.
        VALID_ELEMENTS (set): A set of valid elements (atoms) used for extracting the chemical formula.

    Returns:
        tuple: A tuple containing three lists:
            1. List of lists of atoms (list): A list of lists containing the chemical elements extracted from the formulas of each CIF file.
            2. List of lists of numbers (list): A list of lists containing the corresponding numbers of atoms in the formulas of each CIF file.
            3. List of integers (list): A list containing the 4-digit year values found in the BibTeX information of each CIF file.

    Example:
        cif_files_list = ["/path/to/cif_file1.cif", "/path/to/cif_file2.cif", ...]
        VALID_ELEMENTS = {'H', 'C', 'S', 'N', 'O'}  # Set of valid elements for the formula extraction
        atoms, numbers, years = process_cif_files(cif_files_list, VALID_ELEMENTS)

    Note:
        The function iterates through the list of CIF files and extracts the chemical formula and year information using the process_cif_file function.
        If an error occurs while processing a CIF file, the function prints an error message and continues to the next file.
        The function uses multiprocessing to parallelize the processing of CIF files, providing a progress bar using tqdm to visualize the progress.
    """
    num_processes = 8  # Use the number of available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Create a list of arguments for each CIF file and VALID_ELEMENTS pair
    args_list = [(cif_file_path, VALID_ELEMENTS) for cif_file_path in cif_files_list]

    # Use the process_cif_file function with map method
    results = list(
        tqdm(pool.starmap(process_cif_file, args_list), total=len(args_list))
    )
    pool.close()
    pool.join()

    # Split the results into separate lists
    ATOMS, NUMBERS, YEARS = zip(*results)

    return ATOMS, NUMBERS, YEARS


# In[9]:


import os
import random
from typing import List, Set

import pandas as pd

# Your get_cif_files function here...
# Your process_cif_files function here...


def process_COD_cif_files_main(
    directory_path: str,
    save_path: str,
    VALID_ELEMENTS: set,
    test_run_mode: bool = False,
):
    """
    Processes CIF files in the given directory to extract chemical formulas and year information.

    Args:
        directory_path (str): The path to the directory containing CIF files.
        save_path (str): The path to save the processed data as a CSV file.
        VALID_ELEMENTS (set): A set of valid elements (atoms) used for extracting the chemical formula.
        test_run_mode (bool): If True, a random subset of CIF files will be processed (for testing).

    Note:
        The function uses the get_cif_files function to obtain the list of CIF files in the directory.
        If test_run_mode is True, a random subset of CIF files will be selected for processing.
        The processed data is saved as a CSV file at the specified save_path.
    """

    cif_files_list = get_cif_files(directory_path)

    if test_run_mode:
        random.shuffle(cif_files_list)
        cif_files_list = cif_files_list[:200]

    ATOMS, NUMBERS, YEARS = process_cif_files(cif_files_list, VALID_ELEMENTS)

    datas = []
    for atom, number, year in zip(ATOMS, NUMBERS, YEARS):
        if atom is None or year is None:
            continue
        datas.append([atom, number, year, sum(number), len(atom)])

    df = pd.DataFrame(datas, columns=["atoms", "numbers", "year", "total", "num_types"])
    df.to_csv(save_path, index=False)


# # main.py


directory_path = "./data/raw_data/COD/cod-old/cif"
save_path = "./data/intermediate/PreProcess_COD.csv"
test_run_mode = False
process_COD_cif_files_main(
    directory_path, save_path, VALID_ELEMENTS, test_run_mode=False
)
