import os

import numpy as np
import pytest
from scipy.spatial import KDTree

from src.smoothBranches import removeBranchAnomalies, smoothBranch
from tests.global_fixtures import parameters, base_radius, initial_search_range
from tests.test_utils.comparison import branches_are_equal, subtrees_are_equal
from tests.test_utils.data_import import connect_branches, import_branches, rootPath


@pytest.fixture(scope='module')
def tls_points():
    with open(os.path.join(rootPath(), "TestData/adjustBifurcations/tlsPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


@pytest.fixture(scope="module")
def branches_after_anomaly_removal(tls_points):
    with open(os.path.join(rootPath(), "TestData/smoothBranches/removeBranchAnomalies/result/skeletonPoints.txt")) as f:
        skeleton_points = np.loadtxt(f, dtype=np.double)
    branches = import_branches("TestData/smoothBranches/removeBranchAnomalies/result", skeleton_points,
                               tls_points)
    return branches


class TestRemoveBranchAnomalies:
    @pytest.fixture
    def input_branches_trunk(self):
        with open(os.path.join(rootPath(), "TestData/smoothBranches/removeBranchAnomalies/input/skeletonPoints.txt")) as f:
            skeleton_points = np.loadtxt(f, dtype=np.double)
        branches = import_branches("TestData/smoothBranches/removeBranchAnomalies/input", skeleton_points)
        return connect_branches(branches)[0]


    def test_on_data(self, input_branches_trunk, branches_after_anomaly_removal, parameters):
        removeBranchAnomalies(input_branches_trunk)
        assert subtrees_are_equal(input_branches_trunk, connect_branches(branches_after_anomaly_removal)[0],
                                  compareTlsPoints=False)


class TestSmoothBranch:
    @pytest.fixture
    def result_branches(self):
        with open(os.path.join(rootPath(), "TestData/smoothBranches/smoothBranch/result/skeletonPoints.txt")) as f:
            skeleton_points = np.loadtxt(f, dtype=np.double)
        branches = import_branches("TestData/smoothBranches/smoothBranch/result", skeleton_points)
        return branches


    def test_on_data(self, branches_after_anomaly_removal, result_branches, tls_points, initial_search_range, parameters):
        tlsLookUp = KDTree(tls_points)
        for branch, expected_branch in zip(branches_after_anomaly_removal, result_branches):
            smoothBranch(branch, tlsLookUp, 2 * initial_search_range, parameters["resolution"],
                         np.deg2rad(parameters["smoothAngleThreshold"]))
            assert branches_are_equal(branch, expected_branch, compareTlsPoints=False)
