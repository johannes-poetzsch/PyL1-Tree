import os.path

import numpy as np
from scipy.spatial import KDTree

from tests.global_fixtures import *
from src.L1MedianAlgorithm.generateBranches import (calculateDensity, computeSmoothDirectionalDegrees,
                                                    filterCloseDataPoints, findNewBranches, initializeSkeleton,
                                                    recomputeSkeletonPoints, setFixPoints)


@pytest.fixture
def branches_after_fixed_range():
    data_path = "TestData/L1MedianAlgorithm/fixedRange/newBranches.txt"
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
    if not hasattr(densities, 'value'):
        data_path = "TestData/InputTlsPoints/Densities.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            densities.value = np.loadtxt(input_file, dtype=np.double)
    return densities.value.copy()


@pytest.fixture
def ignored_after_fixed_range():
    if not hasattr(ignored_after_fixed_range, 'value'):
        data_path = "TestData/L1MedianAlgorithm/fixedRange/Ignored.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            ignored_after_fixed_range.value = np.loadtxt(input_file, dtype=np.bool)
    return ignored_after_fixed_range.value.copy()


@pytest.fixture
def ignored_after_second_range():
    if not hasattr(ignored_after_second_range, 'value'):
        data_path = "TestData/L1MedianAlgorithm/secondRange/Ignored.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            ignored_after_second_range.value = np.loadtxt(input_file, dtype=np.bool)
    return ignored_after_second_range.value.copy()


@pytest.fixture
def fixed_after_fixed_range():
    data_path = "TestData/L1MedianAlgorithm/fixedRange/isFix.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def fixed_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/isFix.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.bool)


@pytest.fixture
def is_branch_after_fixed_range():
    if not hasattr(is_branch_after_fixed_range, 'value'):
        data_path = "TestData/L1MedianAlgorithm/fixedRange/isBranch.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            is_branch_after_fixed_range.value = np.loadtxt(input_file, dtype=np.bool)
    return is_branch_after_fixed_range.value.copy()


@pytest.fixture
def positions_after_fixed_range():
    if not hasattr(positions_after_fixed_range, 'value'):
        data_path = "TestData/L1MedianAlgorithm/fixedRange/Positions.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            positions_after_fixed_range.value = np.loadtxt(input_file, dtype=np.double)
    return positions_after_fixed_range.value.copy()


@pytest.fixture
def positions_after_second_range():
    if not hasattr(positions_after_second_range, 'value'):
        data_path = "TestData/L1MedianAlgorithm/secondRange/Positions.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            positions_after_second_range.value = np.loadtxt(input_file, dtype=np.double)
    return positions_after_second_range.value.copy()


@pytest.fixture
def positions_after_one_iteration():
    data_path = "TestData/L1MedianAlgorithm/PositionsAfterOneIteration.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.double)


@pytest.fixture
def smooth_sigma_after_fixed_range():
    data_path = "TestData/L1MedianAlgorithm/fixedRange/smoothSigma.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.double)


@pytest.fixture
def smooth_sigma_after_second_range():
    data_path = "TestData/L1MedianAlgorithm/secondRange/smoothSigma.txt"
    with open(os.path.join(rootPath(), data_path)) as input_file:
        return np.loadtxt(input_file, dtype=np.double)


def test_calculate_density(tls_points, densities, initial_search_range):
    expected_densities = densities
    tlsLookUp = KDTree(tls_points)
    result_densities = calculateDensity(None, tlsLookUp, 2 * initial_search_range)

    # The original L1Tree implementation uses setdiff to remove neighbors with distance 0 (the point itself)
    # from the neighbor distances. Since this is a set operation, it also removes equal distances.
    # The point at index 36272 has two neighbors with the same distance and has thus an incorrect density.
    expected_densities[36272] = result_densities[36272]

    np.testing.assert_array_almost_equal(result_densities, expected_densities)


def test_recompute_skeleton_points_one_interation(base_radius, parameters, initial_search_range,
                                                  seed_points, tls_points, seed_rel_path_positions, densities,
                                                  positions_after_one_iteration):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(seed_points, seed_rel_path_positions)
    recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, initial_search_range, base_radius,
                            parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"], positions_after_one_iteration, decimal=12)


def test_recompute_skeleton_points_fix_range(base_radius, parameters, initial_search_range,
                                             seed_points, tls_points, seed_rel_path_positions, densities,
                                             positions_after_fixed_range):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(seed_points, seed_rel_path_positions)
    for _ in range(parameters["maxIterationNumFixedRange"]):
        recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, initial_search_range, base_radius,
                                parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"], positions_after_fixed_range)


def test_recompute_skeleton_points_second_range(base_radius, parameters, initial_search_range,
                                                tls_points, seed_rel_path_positions, densities,
                                                positions_after_fixed_range, fixed_after_fixed_range,
                                                ignored_after_fixed_range, is_branch_after_fixed_range,
                                                positions_after_second_range):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(positions_after_fixed_range[~ignored_after_fixed_range],
                                      seed_rel_path_positions[~ignored_after_fixed_range])
    skeletonData["isBranch"] = is_branch_after_fixed_range[~ignored_after_fixed_range]
    skeletonData["isFix"] = fixed_after_fixed_range[~ignored_after_fixed_range]

    searchRange = initial_search_range * (1 + parameters["iterationSpeed"])
    for _ in range(parameters["maxIterationNumFixedRange"]):
        recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, searchRange, base_radius,
                                parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"],
                                         positions_after_second_range[~ignored_after_fixed_range])


def test_filter_close_points(positions_after_fixed_range, ignored_after_fixed_range,
                             seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_fixed_range, seed_rel_path_positions)

    filtered_skeleton_data = filterCloseDataPoints(skeletonData, parameters["nearThreshold"])

    np.testing.assert_array_almost_equal(filtered_skeleton_data["Coords"],
                                         positions_after_fixed_range[~ignored_after_fixed_range])


def test_compute_smooth_directional_degrees(positions_after_fixed_range, ignored_after_fixed_range,
                                            smooth_sigma_after_fixed_range, seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_fixed_range[~ignored_after_fixed_range],
                                      seed_rel_path_positions[~ignored_after_fixed_range])

    computeSmoothDirectionalDegrees(skeletonData, parameters["kSmoothSigma"])

    np.testing.assert_array_almost_equal(skeletonData["Sigma"],
                                         smooth_sigma_after_fixed_range[~ignored_after_fixed_range])


def test_set_fix_points(positions_after_fixed_range, ignored_after_fixed_range, fixed_after_fixed_range,
                        smooth_sigma_after_fixed_range, seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_fixed_range[~ignored_after_fixed_range],
                                      seed_rel_path_positions[~ignored_after_fixed_range])
    skeletonData["Sigma"] = smooth_sigma_after_fixed_range[~ignored_after_fixed_range]

    setFixPoints(skeletonData, parameters["sigmaThreshold"])

    np.testing.assert_array_equal(skeletonData["isFix"], fixed_after_fixed_range[~ignored_after_fixed_range])


def test_find_new_branches(positions_after_fixed_range, ignored_after_fixed_range, fixed_after_fixed_range,
                           smooth_sigma_after_fixed_range, seed_rel_path_positions,
                           branches_after_fixed_range, is_branch_after_fixed_range, parameters):
    skeletonData = initializeSkeleton(positions_after_fixed_range[~ignored_after_fixed_range],
                                      seed_rel_path_positions[~ignored_after_fixed_range])
    skeletonData["Sigma"] = smooth_sigma_after_fixed_range[~ignored_after_fixed_range]
    skeletonData["isFix"] = fixed_after_fixed_range[~ignored_after_fixed_range]

    toFilteredIndex = np.empty(len(ignored_after_fixed_range), dtype=np.intp)
    toFilteredIndex[~ignored_after_fixed_range] = np.arange(np.count_nonzero(~ignored_after_fixed_range))

    expected_branches = [toFilteredIndex[b] for b in branches_after_fixed_range]
    expected_branch_points = [skeletonData["Coords"][b] for b in expected_branches]

    branches = findNewBranches(skeletonData, parameters["kTracingThreshold"], parameters["distanceTracingThreshold"],
                               np.deg2rad(parameters["angleTracingThreshold"]), parameters["sigmaTracingThreshold"],
                               parameters["branchSizeThreshold"])

    np.testing.assert_array_almost_equal(skeletonData["isBranch"],
                                         is_branch_after_fixed_range[~ignored_after_fixed_range])

    result_branch_points = [b.skeletonPoints for b in branches]

    # assert len(result_branch_points) == len(expected_branch_points)
    for rb, eb in zip(result_branch_points[:-1], expected_branch_points):
        np.testing.assert_array_almost_equal(rb, eb)
