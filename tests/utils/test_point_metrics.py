import os

import numpy as np
import pytest
from scipy.spatial import KDTree

from src.utils.pointMetrics import calculateDensity
from tests.global_fixtures import base_radius, initial_search_range
from tests.test_utils.data_import import rootPath


class TestCalculateDensity:
    @pytest.fixture
    def tls_points(self):
        with open(os.path.join(rootPath(), "TestData/adjustBifurcations/tlsPoints.txt")) as f:
            return np.loadtxt(f, dtype=np.double)


    @pytest.fixture
    def expected_densities(self):
        with open(os.path.join(rootPath(), "TestData/smoothBranches/smoothBranch/densities.txt")) as f:
            return np.loadtxt(f, dtype=np.double)


    def test_on_data(self, tls_points, expected_densities, initial_search_range):
        densities = calculateDensity(tls_points, KDTree(tls_points), 2 * initial_search_range)
        np.testing.assert_almost_equal(densities, expected_densities, decimal=15)
