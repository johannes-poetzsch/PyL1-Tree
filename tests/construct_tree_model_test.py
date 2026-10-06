import os

import numpy as np
import pytest

from src.constructTreeModel import calculateBranchRadiiCentrifugal, getContinuations
from tests.test_utils.data_import import connect_branches, import_branches, rootPath


@pytest.fixture
def tls_points():
    with open(os.path.join(rootPath(), "TestData/constructTreeModel/calculateBranchRadiiCentrifugal/tlsPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


@pytest.fixture
def branches(tls_points):
    with open(os.path.join(rootPath(), "TestData/constructTreeModel/calculateBranchRadiiCentrifugal/skeletonPoints.txt")) as f:
        skeleton_points = np.loadtxt(f, dtype=np.double)
    return import_branches("TestData/constructTreeModel/calculateBranchRadiiCentrifugal",
                           skeleton_points, tls_points)


@pytest.fixture
def trunk(branches):
    return connect_branches(branches)[0]


@pytest.fixture
def expected_branch_radii(branches):
    with open(os.path.join(rootPath(), "TestData/constructTreeModel/calculateBranchRadiiCentrifugal/radii.txt")) as f:
        radii = np.loadtxt(f, dtype=np.double)
    branchRadii = {}
    for branch, radius in zip(branches, radii):
        branchRadii[branch] = radius
    return branchRadii


@pytest.fixture
def expected_radii_are_credible(branches):
    with open(os.path.join(rootPath(), "TestData/constructTreeModel/calculateBranchRadiiCentrifugal/radiiCredible.txt")) as f:
        credible = np.loadtxt(f, dtype=np.bool)
    branchCredible = {}
    for branch, c in zip(branches, credible):
        branchCredible[branch] = c
    return branchCredible


@pytest.fixture
def is_expected_continuation(branches):
    with open(os.path.join(rootPath(), "TestData/constructTreeModel/calculateBranchRadiiCentrifugal/isContinuation.txt")) as f:
        is_continuation = np.loadtxt(f, dtype=np.bool)
    expected_to_be_continuation = {}
    for branch, is_c in zip(branches, is_continuation):
        expected_to_be_continuation[branch] = is_c
    return expected_to_be_continuation


def test_calculate_branch_radii_centrifugal(trunk, branches, expected_branch_radii, expected_radii_are_credible):
    branch_radii, radii_are_credible = calculateBranchRadiiCentrifugal(trunk)

    for branch in branches:
        assert branch_radii[branch] == pytest.approx(expected_branch_radii[branch])
        assert radii_are_credible[branch] == expected_radii_are_credible[branch]


def test_get_continuations(trunk, branches, expected_branch_radii, is_expected_continuation):
    branchContinuations = getContinuations(trunk, expected_branch_radii)

    for branch in branches:
        is_continuation = False if branch.isStem() else branch is branchContinuations[branch.parent]
        assert is_continuation == is_expected_continuation[branch]
