import numpy as np
import os
import pytest

from src.constructTreeGraph import addBranchEndPoints, preFilterBranches
from src.objects.branch import Branch
from tests.global_fixtures import base_point, base_radius, tls_points
from tests.utils.data_import import import_branches, rootPath


@pytest.fixture
def branches(skeleton_points, tls_points):
    return import_branches("TestData/constructTreeGraph", skeleton_points, tls_points)


@pytest.fixture
def filtered_branches(branches, removed_branch_indices):
    return [b for idx, b in enumerate(branches) if idx not in removed_branch_indices]


@pytest.fixture
def removed_branch_indices(branches):
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/removedBranches.txt")) as f:
        return np.loadtxt(f, dtype=int) - 1


@pytest.fixture
def skeleton_points():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/skeletonPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


def test_point_assignment(branches, tls_points):
    expected_branches = branches
    branches = [Branch(b.skeletonPoints) for b in branches]
    Branch.assignAllPoints(tls_points, branches, useTrimming=True)

    for branch, expected_branch in zip(branches, expected_branches):
        np.testing.assert_array_equal(branch.tlsPoints.sort(axis=0), expected_branch.tlsPoints.sort(axis=0))


def test_pre_filter_branches(base_radius, branches, removed_branch_indices):
    expected_filtered_branches = [b for idx, b in enumerate(branches) if idx not in removed_branch_indices]

    filtered_branches, pruned_points = preFilterBranches(branches, base_radius)

    filtered_branches.sort(key=lambda b: b.id)
    np.testing.assert_array_equal(filtered_branches, expected_filtered_branches)


def test_add_branch_end_points(filtered_branches):
    addBranchEndPoints(filtered_branches)
    pass
