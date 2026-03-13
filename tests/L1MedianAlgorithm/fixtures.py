import numpy as np
import os
import pytest
from scipy.io import loadmat

from tests.global_fixtures import rootPath


@pytest.fixture
def branches_after_first_range():
    data_path = "TestData/L1MedianAlgorithm/firstRange/newBranches.txt"
    branches = []
    with open(os.path.join(rootPath(), data_path)) as input_file:
        for line in input_file:
            branches.append(np.fromstring(line, dtype=int, sep=" ") - 1) # subtract 1 to covert from 1- to 0-indexed
    return branches


@pytest.fixture
def branches_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/newBranches.txt"
    branches = []
    with open(os.path.join(rootPath(), data_path)) as input_file:
        for line in input_file:
            branches.append(np.fromstring(line, dtype=int, sep=" ") - 1) # subtract 1 to covert from 1- to 0-indexed
    return branches


@pytest.fixture
def densities():
    data_path = "TestData/InputTlsPoints/Densities.mat"
    densities = loadmat(os.path.join(rootPath(), data_path))["densities"].squeeze()
    return densities


@pytest.fixture
def fixed_after_first_range():
    data_path = "TestData/L1MedianAlgorithm/firstRange/isFix.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/isFix.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after_finding_branches():
    data_path = "TestData/L1MedianAlgorithm/firstRange/isFixAfterFindingBranches.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after_finding_branches_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/isFixAfterFindingBranches.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def ignored_after_first_range():
    data_path = "TestData/L1MedianAlgorithm/firstRange/Ignored.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def ignored_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/Ignored.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def is_branch_after_first_range():
    data_path = "TestData/L1MedianAlgorithm/firstRange/isBranch.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def is_branch_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/isBranch.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture()
def predecessors():
    if not hasattr(predecessors, 'idx'):
        data_path = "TestData/SeedPoints/GeodesicGraph/Predecessors.txt"
        with open(os.path.join(rootPath(), data_path)) as f:
            predecessors.idx = np.loadtxt(f, dtype=int)
    return predecessors.idx.copy()


@pytest.fixture
def positions_after_first_range():
    data_path = "TestData/L1MedianAlgorithm/firstRange/Positions.mat"
    positions = loadmat(os.path.join(rootPath(), data_path))["positions"]
    return positions


@pytest.fixture
def positions_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/Positions.mat"
    positions = loadmat(os.path.join(rootPath(), data_path))["positions"]
    return positions


@pytest.fixture
def positions_after_one_iteration():
    data_path = "TestData/L1MedianAlgorithm/PositionsAfterOneIteration.mat"
    positions = loadmat(os.path.join(rootPath(), data_path))['positions']
    return positions


@pytest.fixture
def rel_path_positions():
    if not hasattr(rel_path_positions, 'loc'):
        data_path = "TestData/SeedPoints/GeodesicGraph/PathLengths.txt"
        with open(os.path.join(rootPath(), data_path)) as f:
            path_lengths = np.loadtxt(f, dtype=int)
        data_path = "TestData/SeedPoints/GeodesicGraph/PathLocations.txt"
        with open(os.path.join(rootPath(), data_path)) as f:
            path_locations = np.loadtxt(f, dtype=int)
        rel_path_positions.loc = path_locations / path_lengths
    return rel_path_positions.loc.copy()


@pytest.fixture
def seed_points() -> np.ndarray:
    data_path = "TestData/SeedPoints/Positions.mat"
    seed_points = loadmat(os.path.join(rootPath(), data_path))["seeds"]
    return seed_points


@pytest.fixture()
def seed_rel_path_positions():
    if not hasattr(seed_rel_path_positions, 'loc'):
        with open(os.path.join(rootPath(), "TestData/SeedPoints/PathLengths.txt")) as f:
            path_lengths = np.loadtxt(f, dtype=int)
        with open(os.path.join(rootPath(), "TestData/SeedPoints/PathLocations.txt")) as f:
            path_locations = np.loadtxt(f, dtype=int)
        seed_rel_path_positions.loc = path_locations / path_lengths
    return seed_rel_path_positions.loc.copy()


@pytest.fixture
def smooth_sigma_after_first_range():
    data_path = "TestData/L1MedianAlgorithm/firstRange/smoothSigma.mat"
    smooth_sigma = loadmat(os.path.join(rootPath(), data_path))["smoothSigma"].squeeze()
    return smooth_sigma


@pytest.fixture
def smooth_sigma_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/smoothSigma.mat"
    sigma = loadmat(os.path.join(rootPath(), data_path))["sigma"].squeeze()
    return sigma
