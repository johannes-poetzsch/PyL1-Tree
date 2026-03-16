import numpy as np
from scipy.spatial import KDTree

from tests.global_fixtures import base_point, base_radius, initial_search_range, parameters, tls_points
from tests.L1MedianAlgorithm.fixtures import (current_range, densities,
                                              fixed, fixed_after, ignored, ignored_after, is_branch, is_branch_after,
                                              predecessors, positions, positions_after, positions_after_one_iteration,
                                              seed_points, seed_rel_path_positions, search_range, smooth_sigma,
                                              smooth_sigma_after)
from src.L1MedianAlgorithm.generateBranches import (calculateDensity, computeSmoothDirectionalDegrees,
                                                    filterCloseDataPoints, initializeSkeleton,
                                                    recomputeSkeletonPoints, setFixPoints)


def test_calculate_density(tls_points, densities, initial_search_range):
    expected_densities = densities
    tlsLookUp = KDTree(tls_points)
    result_densities = calculateDensity(None, tlsLookUp, 2 * initial_search_range)

    # The original L1Tree implementation uses setdiff to remove neighbors with distance 0 (the point itself)
    # from the neighbor distances. Since this is a set operation, it also removes equal distances.
    # The point at index 36272 has two neighbors with the same distance and has thus an incorrect density.
    expected_densities[36272] = result_densities[36272]

    np.testing.assert_array_almost_equal(result_densities, expected_densities, decimal=15)


def test_recompute_skeleton_points(base_radius, parameters, search_range, tls_points, seed_rel_path_positions, densities,
                                   fixed, is_branch, ignored, positions, positions_after):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(positions[~ignored], seed_rel_path_positions[~ignored])
    skeletonData["isBranch"] = is_branch[~ignored]
    skeletonData["isFix"] = fixed[~ignored]

    for iteration in range(parameters["maxIterationNumFixedRange"]):
        recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, search_range, base_radius,
                                parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"], positions_after[~ignored], decimal=5)


def test_recompute_skeleton_points_one_interation(base_radius, parameters, tls_points, seed_rel_path_positions,
                                                  densities, fixed, ignored, is_branch,
                                                  positions, positions_after_one_iteration, search_range):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(positions[~ignored], seed_rel_path_positions[~ignored])
    skeletonData["isBranch"] = is_branch[~ignored]
    skeletonData["isFix"] = fixed[~ignored]
    recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, search_range, base_radius,
                            parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"], positions_after_one_iteration[~ignored], decimal=12)


def test_filter_close_points(parameters, seed_rel_path_positions,
                             fixed, ignored, ignored_after, is_branch, positions_after):
    skeletonData = initializeSkeleton(positions_after[~ignored], seed_rel_path_positions[~ignored])
    skeletonData["isBranch"] = is_branch[~ignored]
    skeletonData["isFix"] = fixed[~ignored]

    filtered_skeleton_data = filterCloseDataPoints(skeletonData, parameters["nearThreshold"])
    np.testing.assert_array_equal(filtered_skeleton_data["Coords"], positions_after[~ignored_after])


def test_compute_smooth_directional_degrees(parameters, seed_rel_path_positions, fixed, ignored_after, is_branch,
                                            positions_after, smooth_sigma, smooth_sigma_after):
    skeletonData = initializeSkeleton(positions_after[~ignored_after], seed_rel_path_positions[~ignored_after])
    skeletonData["isBranch"] = is_branch[~ignored_after]
    skeletonData["isFix"] = fixed[~ignored_after]
    skeletonData["Sigma"] = smooth_sigma[~ignored_after]

    computeSmoothDirectionalDegrees(skeletonData, parameters["kSmoothSigma"])

    np.testing.assert_array_almost_equal(skeletonData["Sigma"], smooth_sigma_after[~ignored_after], decimal=15)


def test_set_fix_points(parameters, seed_rel_path_positions,
                        fixed, fixed_after, ignored_after, is_branch, positions_after, smooth_sigma_after):
    skeletonData = initializeSkeleton(positions_after[~ignored_after], seed_rel_path_positions[~ignored_after])
    skeletonData["Sigma"] = smooth_sigma_after[~ignored_after]
    skeletonData["isFix"] = fixed[~ignored_after]
    skeletonData["isBranch"] = is_branch[~ignored_after]

    setFixPoints(skeletonData, parameters["sigmaThreshold"])

    np.testing.assert_array_equal(skeletonData["isFix"], fixed_after[~ignored_after])
