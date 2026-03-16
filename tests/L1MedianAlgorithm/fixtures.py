import numpy as np
import os
import pytest
from scipy.io import loadmat

from tests.global_fixtures import rootPath


@pytest.fixture(params=["first Range", "second Range", "final Range"])
def current_range(request):
    return request.param


@pytest.fixture
def branches(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/newBranches.txt"
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/newBranches.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/newBranches.txt"
    else:
        raise ValueError("Invalid range")
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
def fixed(current_range):
    if current_range == "first Range":
        return np.zeros(5086, dtype=bool)
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/isFixAfterFindingBranches.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/Fixed.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/isFix.txt"
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/isFix.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/FixedAfter.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after_finding_branches(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/isFixAfterFindingBranches.txt"
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/isFixAfterFindingBranches.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/FixedAfterBranches.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def ignored(current_range):
    if current_range == "first Range":
        return np.zeros(5086, dtype=bool)
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/Ignored.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/Ignored.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def ignored_after(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/Ignored.txt"
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/Ignored.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/IgnoredAfter.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def is_branch(current_range):
    if current_range == "first Range":
        return np.zeros(5086, dtype=bool)
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/isBranch.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/isBranch.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def is_branch_after(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/isBranch.txt"
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/isBranch.txt"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/isBranchAfter.txt"
    else:
        raise ValueError("Invalid range")
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
def positions(current_range):
    if current_range == "first Range":
        data_path = "TestData/SeedPoints/Positions.mat"
        return loadmat(os.path.join(rootPath(), data_path))["seeds"]
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/Positions.mat"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/Positions.mat"
    else:
        raise ValueError("Invalid range")
    return loadmat(os.path.join(rootPath(), data_path))["positions"]


@pytest.fixture
def positions_after(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/Positions.mat"
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/Positions.mat"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/PositionsAfter.mat"
    else:
        raise ValueError("Invalid range")
    return loadmat(os.path.join(rootPath(), data_path))["positions"]


@pytest.fixture
def positions_after_one_iteration(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/PositionsAfterOneIteration.mat"
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/PositionsAfterOneIteration.mat"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/PositionsAfterOneIteration.mat"
    else:
        raise ValueError("Invalid range")
    return loadmat(os.path.join(rootPath(), data_path))['positions']


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
def search_range(base_radius, current_range, initial_search_range, parameters):
    if current_range == "first Range":
        return initial_search_range
    elif current_range == "second Range":
        return initial_search_range * (1 + parameters["iterationSpeed"])
    elif current_range == "final Range":
        final_search_range = initial_search_range
        while final_search_range * (1 + parameters["iterationSpeed"]) <= 2 * base_radius:
            final_search_range *= (1 + parameters["iterationSpeed"])
        return final_search_range
    else:
        raise ValueError("Invalid range")


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
def smooth_sigma(current_range):
    if current_range == "first Range":
        return np.zeros(5086, dtype=np.double)
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/smoothSigma.mat"
        return loadmat(os.path.join(rootPath(), data_path))["smoothSigma"].squeeze()
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/smoothSigma.mat"
        return loadmat(os.path.join(rootPath(), data_path))["sigma"].squeeze()
    else:
        raise ValueError("Invalid range")


@pytest.fixture
def smooth_sigma_after(current_range):
    if current_range == "first Range":
        data_path = "TestData/L1MedianAlgorithm/firstRange/smoothSigma.mat"
        return loadmat(os.path.join(rootPath(), data_path))["smoothSigma"].squeeze()
    elif current_range == "second Range":
        data_path = "TestData/L1MedianAlgorithm/secondRange/smoothSigma.mat"
    elif current_range == "final Range":
        data_path = "TestData/L1MedianAlgorithm/finalRange/smoothSigmaAfter.mat"
    else:
        raise ValueError("Invalid range")
    return loadmat(os.path.join(rootPath(), data_path))["sigma"].squeeze()
