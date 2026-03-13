import numpy as np

from src.L1MedianAlgorithm.findBranches import findNewBranches
from src.L1MedianAlgorithm.generateBranches import initializeSkeleton
from tests.global_fixtures import parameters
from tests.L1MedianAlgorithm.fixtures import (positions_after_first_range, positions_after_second_range,
                                              ignored_after_first_range, ignored_after_second_range,
                                              fixed_after_first_range, fixed_after_second_range,
                                              smooth_sigma_after_first_range, smooth_sigma_after_second_range,
                                              seed_rel_path_positions,
                                              branches_after_first_range, branches_after_second_range,
                                              is_branch_after_first_range, is_branch_after_second_range,
                                              fixed_after_finding_branches, fixed_after_finding_branches_second_range)


def test_find_new_branches(positions_after_first_range, ignored_after_first_range, fixed_after_first_range,
                           smooth_sigma_after_first_range, seed_rel_path_positions,
                           branches_after_first_range, is_branch_after_first_range,
                           fixed_after_finding_branches, parameters):
    skeletonData = initializeSkeleton(positions_after_first_range[~ignored_after_first_range],
                                      seed_rel_path_positions[~ignored_after_first_range])
    skeletonData["isFix"] = fixed_after_first_range[~ignored_after_first_range]
    skeletonData["Sigma"] = smooth_sigma_after_first_range[~ignored_after_first_range]

    toFilteredIndex = np.empty(len(ignored_after_first_range), dtype=np.intp)
    toFilteredIndex[~ignored_after_first_range] = np.arange(np.count_nonzero(~ignored_after_first_range))

    expected_branches = [toFilteredIndex[b] for b in branches_after_first_range]
    expected_branch_points = [skeletonData["Coords"][b] for b in expected_branches]

    branches = findNewBranches(skeletonData, parameters["kTracingThreshold"], parameters["distanceTracingThreshold"],
                               np.deg2rad(parameters["angleTracingThreshold"]), parameters["sigmaTracingThreshold"],
                               parameters["branchSizeThreshold"])

    np.testing.assert_array_equal(skeletonData["isBranch"], is_branch_after_first_range[~ignored_after_first_range])
    np.testing.assert_array_equal(skeletonData["isFix"], fixed_after_finding_branches[~ignored_after_first_range])

    result_branch_points = [b.skeletonPoints for b in branches]

    assert len(result_branch_points) == len(expected_branch_points)
    for rb, eb in zip(result_branch_points[:-1], expected_branch_points):
        np.testing.assert_array_almost_equal(rb, eb)


def test_find_new_branches_second_range(positions_after_second_range, ignored_after_second_range,
                                        fixed_after_second_range, smooth_sigma_after_second_range,
                                        seed_rel_path_positions, branches_after_second_range,
                                        is_branch_after_first_range, is_branch_after_second_range,
                                        fixed_after_finding_branches_second_range, parameters):
    skeletonData = initializeSkeleton(positions_after_second_range[~ignored_after_second_range],
                                      seed_rel_path_positions[~ignored_after_second_range])
    skeletonData["isBranch"] = is_branch_after_first_range[~ignored_after_second_range]
    skeletonData["isFix"] = fixed_after_second_range[~ignored_after_second_range]
    skeletonData["Sigma"] = smooth_sigma_after_second_range[~ignored_after_second_range]

    toFilteredIndex = np.empty(len(ignored_after_second_range), dtype=np.intp)
    toFilteredIndex[~ignored_after_second_range] = np.arange(np.count_nonzero(~ignored_after_second_range))

    expected_branches = [toFilteredIndex[b] for b in branches_after_second_range]
    expected_branch_points = [skeletonData["Coords"][b] for b in expected_branches]

    branches = findNewBranches(skeletonData, parameters["kTracingThreshold"], parameters["distanceTracingThreshold"],
                               np.deg2rad(parameters["angleTracingThreshold"]), parameters["sigmaTracingThreshold"],
                               parameters["branchSizeThreshold"])

    np.testing.assert_array_equal(skeletonData["isBranch"], is_branch_after_second_range[~ignored_after_second_range])
    np.testing.assert_array_equal(skeletonData["isFix"],
                                  fixed_after_finding_branches_second_range[~ignored_after_second_range])

    result_branch_points = [b.skeletonPoints for b in branches]

    assert len(result_branch_points) == len(expected_branch_points)
    for rb, eb in zip(result_branch_points[:-1], expected_branch_points):
        np.testing.assert_array_almost_equal(rb, eb)
