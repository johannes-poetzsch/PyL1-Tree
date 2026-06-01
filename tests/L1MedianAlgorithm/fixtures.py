import numpy as np
import os
import pytest
from scipy.io import loadmat

from tests.test_utils.data_import import import_branches, rootPath


@pytest.fixture(params=["first Range", "second Range", "final Range"])
def current_range(request):
    return request.param.replace(' ', '')


@pytest.fixture
def branches(current_range, positions_after):
    data_path = r"TestData/L1MedianAlgorithm/" + current_range
    return import_branches(data_path, positions_after)


@pytest.fixture
def densities():
    data_path = "TestData/InputTlsPoints/Densities.mat"
    densities = loadmat(os.path.join(rootPath(), data_path))["densities"].squeeze()
    return densities


@pytest.fixture
def fixed(current_range):
    if current_range == "firstRange":
        return np.zeros(5086, dtype=bool)
    elif current_range == "secondRange":
        data_path = "TestData/L1MedianAlgorithm/firstRange/FixedAfterBranches.txt"
    elif current_range == "finalRange":
        data_path = "TestData/L1MedianAlgorithm/finalRange/Fixed.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after(current_range):
    data_path = "TestData/L1MedianAlgorithm/" + current_range + r"/FixedAfter.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after_finding_branches(current_range):
    data_path = "TestData/L1MedianAlgorithm/" + current_range + r"/FixedAfterBranches.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def ignored(current_range):
    if current_range == "firstRange":
        return np.zeros(5086, dtype=bool)
    elif current_range == "secondRange":
        data_path = "TestData/L1MedianAlgorithm/firstRange/IgnoredAfter.txt"
    elif current_range == "finalRange":
        data_path = "TestData/L1MedianAlgorithm/finalRange/Ignored.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def ignored_after(current_range):
    data_path = "TestData/L1MedianAlgorithm/" + current_range + r"/IgnoredAfter.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def is_branch(current_range):
    if current_range == "firstRange":
        return np.zeros(5086, dtype=bool)
    elif current_range == "secondRange":
        data_path = "TestData/L1MedianAlgorithm/firstRange/isBranchAfter.txt"
    elif current_range == "finalRange":
        data_path = "TestData/L1MedianAlgorithm/finalRange/isBranch.txt"
    else:
        raise ValueError("Invalid range")
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def is_branch_after(current_range):
    data_path = "TestData/L1MedianAlgorithm/" + current_range + r"/isBranchAfter.txt"
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
    if current_range == "firstRange":
        data_path = "TestData/SeedPoints/Positions.mat"
        return loadmat(os.path.join(rootPath(), data_path))["seeds"]
    elif current_range == "secondRange":
        data_path = "TestData/L1MedianAlgorithm/firstRange/PositionsAfter.mat"
    elif current_range == "finalRange":
        data_path = "TestData/L1MedianAlgorithm/finalRange/Positions.mat"
    else:
        raise ValueError("Invalid range")
    return loadmat(os.path.join(rootPath(), data_path))["positions"]


@pytest.fixture
def positions_after(current_range):
    data_path = "TestData/L1MedianAlgorithm/" + current_range + r"/PositionsAfter.mat"
    return loadmat(os.path.join(rootPath(), data_path))["positions"]


@pytest.fixture
def positions_after_one_iteration(current_range):
    data_path = "TestData/L1MedianAlgorithm/" + current_range + r"/PositionsAfterOneIteration.mat"
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
    if current_range == "firstRange":
        return initial_search_range
    elif current_range == "secondRange":
        return initial_search_range * (1 + parameters["iterationSpeed"])
    elif current_range == "finalRange":
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
def sigma(current_range):
    if current_range == "firstRange":
        return np.zeros(5086, dtype=np.double)
    else:
        data_path = "TestData/L1MedianAlgorithm/" + current_range + r"/Sigma.mat"
    return loadmat(os.path.join(rootPath(), data_path))["sigma"].squeeze()


@pytest.fixture
def sigma_after(current_range):
    if current_range == "firstRange":
        return np.zeros(5086, dtype=np.double)
    elif current_range == "secondRange":
        data_path = "TestData/L1MedianAlgorithm/firstRange/smoothSigma.mat"
        return loadmat(os.path.join(rootPath(), data_path))["smoothSigma"].squeeze()
    elif current_range == "finalRange":
        data_path = "TestData/L1MedianAlgorithm/finalRange/smoothSigma.mat"
        return loadmat(os.path.join(rootPath(), data_path))["sigma"].squeeze()
    else:
        raise ValueError("Invalid range")


@pytest.fixture
def smooth_sigma_after(current_range):
    if current_range == "firstRange":
        data_path = "TestData/L1MedianAlgorithm/firstRange/smoothSigma.mat"
        return loadmat(os.path.join(rootPath(), data_path))["smoothSigma"].squeeze()
    elif current_range == "secondRange":
        data_path = "TestData/L1MedianAlgorithm/secondRange/smoothSigma.mat"
    elif current_range == "finalRange":
        data_path = "TestData/L1MedianAlgorithm/finalRange/smoothSigmaAfter.mat"
    else:
        raise ValueError("Invalid range")
    return loadmat(os.path.join(rootPath(), data_path))["sigma"].squeeze()
