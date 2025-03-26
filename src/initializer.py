import ast
import os
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


def load_training_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from the specified npz file.

    Args:
        data_path (str): Path to the npz file.

    Returns:
        np.ndarray: x_train data.
        np.ndarray: y_train data.
    """
    data = np.load(data_path)
    x_train = data["x_train"]
    y_train = data["y_train"]

    return x_train, y_train


def atom_distribution_from_train(x_train: np.ndarray) -> np.ndarray:
    # Calculate the frequency of values greater than 0 for each column (axis=0)
    atom_frequency = np.sum(x_train > 0, axis=0)
    atom_prob = atom_frequency / np.sum(atom_frequency)
    return atom_prob


def generate_mixed_init_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_mix: int,
    min_tc: float,
    mix_alpha: float,
    batch_size: int,
    rng=np.random.default_rng(1234),
) -> np.ndarray:
    """
    Generate mixed data based on the given conditions.

    Args:
        x_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        num_mix (int): Number of data points to mix.
        min_tc (float): Minimum threshold for y_train values.
        alpha (float): Mixing parameter.
        batch_size (int): Number of mixed samples to generate.

    Returns:
        np.ndarray: Mixed data.
    """
    assert num_mix in [1, 2]
    assert min_tc > 0
    assert batch_size > 0

    # print("デバッグモード")
    # return x_train[y_train >= 10][:batch_size]

    if num_mix == 1:
        print("num_mix is 1, so return random sampled x_train")
    # Extract x_train samples where y_train is above the given threshold
    x_filtered = x_train[y_train >= min_tc]

    # Randomly select num_mix * batch_size samples from the filtered data without replacement
    if x_filtered.shape[0] < num_mix * batch_size:
        replace = True
    else:
        replace = False
    random_indices = rng.choice(
        x_filtered.shape[0], num_mix * batch_size, replace=replace
    ).reshape(batch_size, num_mix)
    random_samples = x_filtered[random_indices]

    # Sum the samples and normalize to ensure the total is 1
    if mix_alpha > 0:
        ratio = np.random.beta(mix_alpha, mix_alpha, size=(batch_size, 1))
    else:
        ratio = np.ones((batch_size, 1))
    ratio = np.concatenate([ratio, 1 - ratio], axis=1)
    mixed_data = np.sum(random_samples * ratio[..., np.newaxis], axis=1)
    mixed_data /= np.sum(mixed_data, axis=1)[:, np.newaxis]

    assert mixed_data.shape == (batch_size, x_train.shape[1])
    assert (np.abs(np.sum(mixed_data, axis=1) - 1.0) <= 1e-10).all()

    return mixed_data


# Run the test function to validate the changes
def mutate_init_data(
    init_data: np.ndarray,
    atom_prob: np.ndarray,
    mutate_prob: float,
    rng=np.random.default_rng(1234),
) -> np.ndarray:
    """
    Mutate the init_data based on the given atom probability distribution with a defined mutation probability.

    Args:
        init_data (np.ndarray): Input data to be mutated.
        atom_prob (np.ndarray): Atom probability distribution.
        mutate_prob (float): Probability of mutation for each index.

    Returns:
        np.ndarray: Mutated data.
    """
    if mutate_prob < 0:
        print("no mutation")
        return init_data

    mutated_data = init_data.copy()
    rows, current_indices = np.where(mutated_data > 0)
    values = init_data[np.where(mutated_data > 0)]

    # Determine the number of mutations based on mutate_prob
    mutate_flags = rng.random(len(rows)) < mutate_prob

    # Get the indices to mutate
    rows_to_mutate = rows[mutate_flags]
    indices_to_mutate = current_indices[mutate_flags]

    # Sample new indices based on the atom probability distribution
    new_indices = rng.choice(len(atom_prob), size=len(rows_to_mutate), p=atom_prob)

    # Update the original data with the new indices
    mutated_data[rows_to_mutate, indices_to_mutate] = 0
    mutated_data[rows_to_mutate, new_indices] = (
        mutated_data[rows_to_mutate, new_indices] + values[mutate_flags]
    )  # Set the values to the new indices

    return mutated_data


def perturb_init_data(
    init_data: np.ndarray,
    atom_prob: np.ndarray,
    purterb_val: float,
    purterb_prob: float,
    rng=np.random.default_rng(1234),
) -> np.ndarray:
    """
    Vectorized version of perturbing the given mixed data based on atom probability and perturbation parameters.
    This function perturbs the whole array at once.

    Args:
        init_data (np.ndarray): Input data to be perturbed.
        atom_prob (np.ndarray): Atom probability distribution.
        purterb_val (float): Value to be added to the init_data at the perturbed index.
        purterb_prob (float): Probability of perturbing a given element in init_data.

    Returns:
        np.ndarray: Perturbed mixed data.
    """
    if purterb_prob < 0:
        print("no perturbation")
        return init_data
    # Determine which elements will be perturbed
    perturb_flags = (rng.random(init_data.shape) < purterb_prob).astype(int)
    noise_array = rng.random(init_data.shape) * purterb_val
    noise_array = noise_array * perturb_flags

    # Use advanced indexing to add purterb_val to the appropriate positions in init_data
    perturb_data = init_data.copy()
    perturb_data += noise_array
    perturb_data = perturb_data / np.sum(perturb_data, axis=1)[:, np.newaxis]

    return perturb_data


def test_mutate_init_data(atom_prob: np.ndarray) -> None:
    """
    Test the effect of mutation on dummy data and validate against the given atom probability distribution.

    Args:
        atom_prob (np.ndarray)): Atom probability distribution.

    Raises:
        AssertionError: If the difference between generated probability and atom_prob is greater than 1E-2.

    Returns:
        None: This function will print the results and raise assertions if tests fail.
    """
    count_array = np.zeros((118,)).astype(np.float32)
    dummy_input = np.ones((1000, 118)).astype(np.float32)
    print("testgin mutate_init_data...", end="")
    for _ in tqdm(range(500)):
        count_array += mutate_init_data(dummy_input, atom_prob, mutate_prob=1.0).sum(
            axis=0
        )

    prob = count_array / count_array.sum()
    assert (prob - atom_prob < 1e-2).all()
    print("OK")


def test_generate_mixed_init_data():
    """
    Test the generate_mixed_data function with specific test data.
    """
    print("testing generate_mixed_init_data...", end=" ")
    # Test data for the function
    test_x_train = np.array([[0, 0.5, 0.5], [1.0, 0, 0], [0, 0, 1]])
    test_y_train = np.array([5, 5, 1])

    # Test the generate_mixed_data function with the provided parameters
    test_mixed_data = generate_mixed_init_data(
        test_x_train, test_y_train, num_mix=2, min_tc=2, batch_size=1, mix_alpha=-1.0
    )

    expected_result = np.array([[0.5, 0.25, 0.25]])
    # assert np.array_equal(test_mixed_data, expected_result), f"Expected {expected_result} but got {test_mixed_data}"
    print("OK")


def process_and_perturb_data(
    data_path: str,
    mixed_data_path: str,
    purterb_val: float,
    purterb_prob: float,
    mutate_prob: float,
    num_mix: int,
    min_tc: int,
    mix_alpha: float,
    batch_size: int,
    fixed_mixed_init_data_mode: bool,
    test: bool = False,
) -> np.ndarray:
    """
    Process, mutate, and perturb data for inverse problem initial solution candidates based on given parameters.

    This function performs a series of operations on data loaded from the given path. The primary steps include:

    1. Loading the training data.
    2. Generating or loading mixed data, which involves combining different data points based on specific conditions.
    3. Mutating the mixed data based on atom probability distribution and a given mutation probability.
    4. Perturbing the mutated data by introducing minor changes at random positions.

    The function offers flexibility in terms of whether to generate mixed data afresh or load pre-existing mixed data.
    Additionally, it provides an option to run test functions to validate the underlying operations.
    Args:
        data_path (str): Path to the training data.
        mixed_data_path (str): Path to the saved mixed data.
        purterb_val (float): Value to perturb the data.
        purterb_prob (float): Probability of perturbing a given data point. -1 means no perturbation.
        mutate_prob (float): Probability of mutating the data. -1 means no mutation.
        num_mix (int): Number of data points to mix.
        min_tc (int): Minimum threshold for y_train values.
        batch_size (int): Number of mixed samples to generate.
        fixed_mixed_init_data_mode (bool): Flag to determine if mixed data is loaded from file or generated.
        test (bool, optional): If set to True, runs test functions. Defaults to False.

    Returns:
        np.ndarray: Perturbed data.
    """
    # Load training data
    x_train, y_train = load_training_data(data_path)
    atom_prob = atom_distribution_from_train(x_train)

    # Run test functions if the test flag is set
    if test:
        test_mutate_init_data(atom_prob)
        test_generate_mixed_init_data()

    # Generate or load mixed data based on the flag
    if fixed_mixed_init_data_mode == True:
        if os.path.exists(mixed_data_path):
            mixed_data = np.load(mixed_data_path)
        else:
            mixed_data = generate_mixed_init_data(
                x_train, y_train, num_mix, min_tc, mix_alpha, batch_size
            )
            os.makedirs(os.path.dirname(mixed_data_path), exist_ok=True)
            np.save(mixed_data_path, mixed_data)
    elif fixed_mixed_init_data_mode == "rewrite":
        mixed_data = generate_mixed_init_data(
            x_train, y_train, num_mix, min_tc, mix_alpha, batch_size
        )
        os.makedirs(os.path.dirname(mixed_data_path), exist_ok=True)
        np.save(mixed_data_path, mixed_data)
    elif fixed_mixed_init_data_mode == False:
        mixed_data = generate_mixed_init_data(
            x_train, y_train, num_mix, min_tc, mix_alpha, batch_size
        )
    else:
        raise ValueError(
            "Invalid fixed_mixed_init_data_mode. Expected True, False, or 'rewrite'"
        )

    # Mutate and perturb the data
    mutated_data = mutate_init_data(mixed_data, atom_prob, mutate_prob)
    perturbed_data = perturb_init_data(
        mutated_data, atom_prob, purterb_val, purterb_prob
    )

    return perturbed_data


def generate_dataset_based_init_candidates(cfg: DictConfig, test: bool) -> np.ndarray:
    """
    Generate an initial tensor for inverse problem solution based on the provided configuration.
    The generated initial candidates are based on training dataset and the given configuration.

    Args:
        cfg (Any): Configuration object that contains the necessary parameters and paths.
        test(bool): If set to True, runs test.

    Returns:
        np.ndarray: Initial tensor after processing and perturbation. shape=(batch_size, cfg.general.type_of_atoms)
    """
    # Extract parameters from the configuration
    dataset_name = cfg.dataset.dataset_name.replace(".npz", "")
    num_mix = cfg.inverse_problem.initialization.num_mix
    purterb_prob = cfg.inverse_problem.initialization.purterb_prob
    mutate_prob = cfg.inverse_problem.initialization.mutate_prob
    purterb_val = cfg.inverse_problem.initialization.purterb_val
    min_tc = cfg.inverse_problem.initialization.min_tc
    mix_alpha = cfg.inverse_problem.initialization.mix_alpha
    batch_size = cfg.inverse_problem.method_parameters.iv_batch_size

    fixed_mixed_init_data_mode = (
        cfg.inverse_problem.initialization.fixed_mixed_init_data_mode
    )
    data_path = os.path.join(cfg.general.master_dir, cfg.dataset.dataset_name)
    mixed_data_path = os.path.join(
        cfg.general.master_dir,
        f"data/tmp/{dataset_name}--bs{batch_size}--mixed{num_mix}-alpha{mix_alpha}.npy",
    )
    # Generate the initial tensor using the `process_and_perturb_data` function
    init_tensor = process_and_perturb_data(
        data_path,
        mixed_data_path,
        purterb_val,
        purterb_prob,
        mutate_prob,
        num_mix,
        min_tc,
        mix_alpha,
        batch_size,
        fixed_mixed_init_data_mode,
        test,
    )

    # Convert the tensor to float32 type
    init_tensor = init_tensor.astype(np.float32)

    return init_tensor


def generate_random_based_init_candidates(
    cfg: DictConfig, batch_size: int, rng=np.random.default_rng(1234)
) -> np.ndarray:
    """
    Generate an initial tensor using random values based on the provided configuration.
    The generated initial candidates are based on random values and the given configuration.

    Args:
        cfg (DictConfig): Configuration object that contains the necessary parameters and paths.
        batch_size (int): Number of samples to generate.

    Returns:
        np.ndarray: Initial tensor with random values.
    """
    # Extract the number of types of atoms from the configuration
    type_of_atoms = cfg.general.type_of_atoms

    # Generate a random tensor of shape (batch_size, type_of_atoms)
    init_tensor = rng.random((batch_size, type_of_atoms))

    # Convert the tensor to float32 type
    init_tensor = init_tensor.astype(np.float32)

    return init_tensor


def generate_reference_based_candidates(cfg: DictConfig, batch_size: int, test: bool):
    type_of_atoms = cfg.general.type_of_atoms
    reference_vector = ast.literal_eval(cfg.inverse_problem.initialization.initializer)
    reference_candidates = np.array(reference_vector)
    assert reference_candidates.shape[0] == type_of_atoms
    batched_reference_candidates = np.tile(reference_candidates, (batch_size, 1))

    # Mutate and perturb the data
    ## Extract parameters from the configuration
    purterb_prob = cfg.inverse_problem.initialization.purterb_prob
    mutate_prob = cfg.inverse_problem.initialization.mutate_prob
    purterb_val = cfg.inverse_problem.initialization.purterb_val
    print("perturb_prob", purterb_prob)
    print("mutate_prob", mutate_prob)
    print("perturb_val", purterb_val)

    ## Extract atomic distribution from the training data
    dataset_name = cfg.dataset.dataset_name.replace(".npz", "")
    data_path = os.path.join(cfg.general.master_dir, cfg.dataset.dataset_name)
    x_train, _ = load_training_data(data_path)
    atom_prob = atom_distribution_from_train(x_train)
    ## mutate and perturb the data
    mutated_data = mutate_init_data(
        batched_reference_candidates, atom_prob, mutate_prob
    )
    perturbed_data = perturb_init_data(
        mutated_data, atom_prob, purterb_val, purterb_prob
    )

    return perturbed_data


def initialize_solution_candidates(
    cfg: DictConfig, batch_size: int, test: bool
) -> np.ndarray:
    """
    Generate an initial tensor for the inverse problem solution based on the provided configuration.

    The function determines which initializer to use (dataset-based or random-based) from the configuration
    and then generates the initial tensor accordingly.

    Args:
        cfg (DictConfig): Configuration object containing necessary parameters.
        batch_size (int): Number of samples to generate.
        test (bool, optional): If set to True, runs test functions when dataset-based initialization is used.
                               Defaults to False.

    Returns:
        np.ndarray: Generated initial tensor.
    """

    initializer = cfg.inverse_problem.initialization.initializer

    if initializer == "dataset":
        return generate_dataset_based_init_candidates(cfg, test=test)
    elif initializer == "random":
        return generate_random_based_init_candidates(cfg, batch_size)
    elif initializer == "specified":
        init_candidates = np.array(
            cfg.inverse_problem.initialization.specified_init_candidates
        )
        print("specified init candidates:")
        assert len(init_candidates.shape) == 2
        assert (
            init_candidates.shape[0]
            == cfg.inverse_problem.method_parameters.iv_batch_size
        )
        return init_candidates
    elif initializer.startswith("["):
        return generate_reference_based_candidates(cfg, batch_size, test=test)
    else:
        raise ValueError(
            f"Unknown initializer: {initializer}. Supported initializers are 'dataset' and 'random'."
        )
