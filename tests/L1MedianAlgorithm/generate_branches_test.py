import numpy as np
from scipy.spatial import KDTree

from tests.global_fixtures import base_point, base_radius, initial_search_range, parameters, tls_points
from tests.L1MedianAlgorithm.fixtures import (densities, fixed_after_first_range, fixed_after_second_range,
                                              fixed_after_finding_branches, ignored_after_first_range,
                                              ignored_after_second_range, is_branch_after_first_range, predecessors,
                                              positions_after_first_range, positions_after_second_range,
                                              positions_after_one_iteration, seed_points,
                                              seed_rel_path_positions, smooth_sigma_after_first_range,
                                              smooth_sigma_after_second_range)
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


def test_recompute_skeleton_points_one_interation(base_radius, parameters, initial_search_range,
                                                  seed_points, tls_points, seed_rel_path_positions, densities,
                                                  positions_after_one_iteration):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(seed_points, seed_rel_path_positions)
    recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, initial_search_range, base_radius,
                            parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"], positions_after_one_iteration, decimal=12)


def test_recompute_skeleton_points_first_range(base_radius, parameters, initial_search_range,
                                               seed_points, tls_points, seed_rel_path_positions, densities,
                                               positions_after_first_range):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(seed_points, seed_rel_path_positions)
    for _ in range(parameters["maxIterationNumFixedRange"]):
        recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, initial_search_range, base_radius,
                                parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"], positions_after_first_range)


def test_recompute_skeleton_points_second_range(base_radius, parameters, initial_search_range,
                                                tls_points, seed_rel_path_positions, densities,
                                                positions_after_first_range, fixed_after_finding_branches,
                                                ignored_after_first_range, is_branch_after_first_range,
                                                positions_after_second_range):
    tlsLookUp = KDTree(tls_points)
    skeletonData = initializeSkeleton(positions_after_first_range[~ignored_after_first_range],
                                      seed_rel_path_positions[~ignored_after_first_range])
    skeletonData["isBranch"] = is_branch_after_first_range[~ignored_after_first_range]
    skeletonData["isFix"] = fixed_after_finding_branches[~ignored_after_first_range]

    searchRange = initial_search_range * (1 + parameters["iterationSpeed"])
    for _ in range(parameters["maxIterationNumFixedRange"]):
        recomputeSkeletonPoints(skeletonData, tlsLookUp, densities, searchRange, base_radius,
                                parameters["minMu"], parameters["maxMu"])

    np.testing.assert_array_almost_equal(skeletonData["Coords"],
                                         positions_after_second_range[~ignored_after_first_range], decimal=5)


def test_filter_close_points(positions_after_first_range, ignored_after_first_range,
                             seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_first_range, seed_rel_path_positions)

    filtered_skeleton_data = filterCloseDataPoints(skeletonData, parameters["nearThreshold"])

    np.testing.assert_array_equal(filtered_skeleton_data["Coords"],
                                  positions_after_first_range[~ignored_after_first_range])


def test_filter_close_points_second_range(positions_after_second_range, ignored_after_first_range,
                                          ignored_after_second_range, is_branch_after_first_range,
                                          fixed_after_finding_branches, seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_second_range[~ignored_after_first_range],
                                      seed_rel_path_positions[~ignored_after_first_range])
    skeletonData["isBranch"] = is_branch_after_first_range[~ignored_after_first_range]
    skeletonData["isFix"] = fixed_after_finding_branches[~ignored_after_first_range]

    filtered_skeleton_data = filterCloseDataPoints(skeletonData, parameters["nearThreshold"])
    np.testing.assert_array_equal(filtered_skeleton_data["Coords"],
                                  positions_after_second_range[~ignored_after_second_range])



def test_compute_smooth_directional_degrees(positions_after_first_range, ignored_after_first_range,
                                            smooth_sigma_after_first_range, seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_first_range[~ignored_after_first_range],
                                      seed_rel_path_positions[~ignored_after_first_range])

    computeSmoothDirectionalDegrees(skeletonData, parameters["kSmoothSigma"])

    np.testing.assert_array_almost_equal(skeletonData["Sigma"],
                                         smooth_sigma_after_first_range[~ignored_after_first_range], decimal=15)


def test_compute_smooth_directional_degrees_second_range(positions_after_second_range, ignored_after_second_range,
                                                         smooth_sigma_after_first_range, smooth_sigma_after_second_range,
                                                         seed_rel_path_positions, fixed_after_finding_branches,
                                                         is_branch_after_first_range, parameters):
    skeletonData = initializeSkeleton(positions_after_second_range[~ignored_after_second_range],
                                      seed_rel_path_positions[~ignored_after_second_range])
    skeletonData["isBranch"] = is_branch_after_first_range[~ignored_after_second_range]
    skeletonData["isFix"] = fixed_after_finding_branches[~ignored_after_second_range]
    skeletonData["Sigma"] = smooth_sigma_after_first_range[~ignored_after_second_range]

    computeSmoothDirectionalDegrees(skeletonData, parameters["kSmoothSigma"])

    np.testing.assert_array_almost_equal(skeletonData["Sigma"],
                                         smooth_sigma_after_second_range[~ignored_after_second_range], decimal=15)


def test_set_fix_points(positions_after_first_range, ignored_after_first_range, fixed_after_first_range,
                        smooth_sigma_after_first_range, seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_first_range[~ignored_after_first_range],
                                      seed_rel_path_positions[~ignored_after_first_range])
    skeletonData["Sigma"] = smooth_sigma_after_first_range[~ignored_after_first_range]

    setFixPoints(skeletonData, parameters["sigmaThreshold"])

    np.testing.assert_array_equal(skeletonData["isFix"], fixed_after_first_range[~ignored_after_first_range])


def test_set_fix_points_second_range(positions_after_second_range, fixed_after_finding_branches,
                                     fixed_after_second_range, ignored_after_second_range, is_branch_after_first_range,
                                     smooth_sigma_after_second_range, seed_rel_path_positions, parameters):
    skeletonData = initializeSkeleton(positions_after_second_range[~ignored_after_second_range],
                                      seed_rel_path_positions[~ignored_after_second_range])
    skeletonData["Sigma"] = smooth_sigma_after_second_range[~ignored_after_second_range]
    skeletonData["isFix"] = fixed_after_finding_branches[~ignored_after_second_range]
    skeletonData["isBranch"] = is_branch_after_first_range[~ignored_after_second_range]

    setFixPoints(skeletonData, parameters["sigmaThreshold"])

    np.testing.assert_array_equal(skeletonData["isFix"], fixed_after_second_range[~ignored_after_second_range])
