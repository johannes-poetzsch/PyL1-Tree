import numpy as np
import pytest
from scipy.spatial import KDTree

from src.L1MedianAlgorithm.generateSeeds import buildNeighborGraph, findBase, generateSeeds, getPathData, \
    removeUnconnectedComponents
from tests.global_fixtures import base_point, base_radius, parameters, tls_points
from tests.L1MedianAlgorithm.fixtures import predecessors, rel_path_positions, seed_points, seed_rel_path_positions


def test_find_base(tls_points, base_point, base_radius):
    expected_base_point = base_point
    expected_base_radius = base_radius

    result_base_point, result_base_radius = findBase(tls_points[:-1])

    assert result_base_radius == pytest.approx(expected_base_radius)
    np.testing.assert_array_almost_equal(result_base_point, expected_base_point, decimal=12)


def test_generate_seeds(base_point, tls_points, seed_points, seed_rel_path_positions, parameters):
    expected_seedPoints, expected_rel_path_positions = seed_points, seed_rel_path_positions
    result_seed_points, result_rel_path_positions, tlsPoints = generateSeeds(tls_points[:-1], parameters)

    np.testing.assert_array_almost_equal(result_seed_points[-1], base_point, decimal=12)
    np.testing.assert_array_equal(result_seed_points[:-1], expected_seedPoints[:-1])
    np.testing.assert_array_equal(result_rel_path_positions, expected_rel_path_positions)


def test_get_path_data(tls_points, predecessors, rel_path_positions):
    expected_predecessors = predecessors
    expected_rel_locations = rel_path_positions

    base_point_idx = len(tls_points) - 1
    tls_points, base_point_idx = removeUnconnectedComponents(tls_points, base_point_idx)

    neighborGraph = buildNeighborGraph(KDTree(tls_points))
    isConnected, result_rel_path_positions, result_predecessors = getPathData(neighborGraph, base_point_idx)

    result_predecessors = np.delete(result_predecessors, base_point_idx)

    assert np.all(isConnected)
    np.testing.assert_array_equal(result_predecessors, expected_predecessors)
    np.testing.assert_array_equal(result_rel_path_positions, expected_rel_locations)
