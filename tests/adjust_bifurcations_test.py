import os

import numpy as np
import pytest

from src.adjustBifurcations import adjustBifurcations
from tests.global_fixtures import parameters
from tests.test_utils.comparison import subtrees_are_equal
from tests.test_utils.data_import import connect_branches, import_branches, rootPath


@pytest.fixture
def tls_points():
    with open(os.path.join(rootPath(), "TestData/adjustBifurcations/tlsPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


@pytest.fixture
def input_branches_trunk(tls_points):
    with open(os.path.join(rootPath(), "TestData/adjustBifurcations/input/skeletonPoints.txt")) as f:
        skeleton_points = np.loadtxt(f, dtype=np.double)
    branches = import_branches("TestData/adjustBifurcations/input", skeleton_points, tls_points)
    return connect_branches(branches)[0]


@pytest.fixture
def result_branches_trunk():
    with open(os.path.join(rootPath(), "TestData/adjustBifurcations/result/skeletonPoints.txt")) as f:
        skeleton_points = np.loadtxt(f, dtype=np.double)
    branches = import_branches("TestData/adjustBifurcations/result", skeleton_points)
    return connect_branches(branches)[0]


def test_on_data(input_branches_trunk, result_branches_trunk, parameters):
    adjustBifurcations(input_branches_trunk, parameters)
    assert subtrees_are_equal(input_branches_trunk, result_branches_trunk, compareTlsPoints=False)
