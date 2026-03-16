import numpy as np

from src.L1MedianAlgorithm.findBranches import findNewBranches
from src.L1MedianAlgorithm.generateBranches import initializeSkeleton
from tests.global_fixtures import parameters
from tests.L1MedianAlgorithm.fixtures import (branches, current_range, fixed_after, fixed_after_finding_branches,
                                              ignored_after, is_branch, is_branch_after, positions_after,
                                              seed_rel_path_positions, smooth_sigma_after)


def test_find_new_branches(branches, parameters, seed_rel_path_positions,
                           fixed_after, fixed_after_finding_branches, ignored_after, is_branch, is_branch_after,
                           positions_after, smooth_sigma_after):
    skeletonData = initializeSkeleton(positions_after[~ignored_after],
                                      seed_rel_path_positions[~ignored_after])
    skeletonData["isBranch"] = is_branch[~ignored_after]
    skeletonData["isFix"] = fixed_after[~ignored_after]
    skeletonData["Sigma"] = smooth_sigma_after[~ignored_after]

    toFilteredIndex = np.empty(len(ignored_after), dtype=np.intp)
    toFilteredIndex[~ignored_after] = np.arange(np.count_nonzero(~ignored_after))

    expected_branches = [toFilteredIndex[b] for b in branches]
    expected_branch_points = [skeletonData["Coords"][b] for b in expected_branches]

    branches = findNewBranches(skeletonData, parameters["kTracingThreshold"], parameters["distanceTracingThreshold"],
                               np.deg2rad(parameters["angleTracingThreshold"]), parameters["sigmaTracingThreshold"],
                               parameters["branchSizeThreshold"])

    np.testing.assert_array_equal(skeletonData["isBranch"], is_branch_after[~ignored_after])
    np.testing.assert_array_equal(skeletonData["isFix"], fixed_after_finding_branches[~ignored_after])

    result_branch_points = [b.skeletonPoints for b in branches]

    assert len(result_branch_points) == len(expected_branch_points)
    for rb, eb in zip(result_branch_points[:-1], expected_branch_points):
        np.testing.assert_array_almost_equal(rb, eb)
