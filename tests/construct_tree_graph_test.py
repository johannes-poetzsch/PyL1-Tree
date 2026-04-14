from itertools import combinations
import numpy as np
import os
import pytest

from src.constructTreeGraph import constructComponents, createSkeletonGraph, preFilterBranches, splitCrookedBranches
from src.objects.branch import Branch
from tests.connect_components_test import unassigned_points
from tests.global_fixtures import base_point, base_radius, parameters, tls_points
from tests.utils.comparison import branches_are_equal, subtrees_are_equal
from tests.utils.data_import import import_branches, rootPath


@pytest.fixture
def branches(skeleton_points, tls_points):
    return import_branches("TestData/constructTreeGraph", skeleton_points, tls_points)


@pytest.fixture
def components(branch_skeleton_points):
    branches = import_branches("TestData/constructTreeGraph/components", branch_skeleton_points)
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/components/componentIDs.txt")) as f:
        component_ids =  np.loadtxt(f, dtype=int) - 1

    component_stems = []
    for component_id in np.unique(component_ids):
        component_branches = [b for b, cid in zip(branches, component_ids) if cid == component_id]
        component_stems.append(component_branches[0])
        start_points = np.array([b[0] for b in component_branches])
        end_points = np.array([b[-1] for b in component_branches])
        for i, j in combinations(range(len(component_branches)), 2):
            if np.all(end_points[i] == start_points[j]):
                component_branches[j].setParent(component_branches[i])
            elif np.all(end_points[j] == start_points[i]):
                component_branches[i].setParent(component_branches[j])
    return component_stems


@pytest.fixture
def end_point_ids():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/endPointIds.txt")) as f:
        return np.loadtxt(f, dtype=int) - 1


@pytest.fixture
def filtered_branches(branches, removed_branch_indices):
    return [b for idx, b in enumerate(branches) if idx not in removed_branch_indices]


@pytest.fixture
def predecessors():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/predecessors.txt")) as f:
        return np.loadtxt(f, dtype=int) - 1


@pytest.fixture
def removed_branch_indices(branches):
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/removedBranches.txt")) as f:
        return np.loadtxt(f, dtype=int) - 1


@pytest.fixture
def skeleton_points():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/skeletonPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


@pytest.fixture
def branch_skeleton_points():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/branchSkeletonPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


def test_point_assignment(branches, tls_points):
    expected_branches = branches
    branches = [Branch(b.skeletonPoints) for b in branches]
    unassigned_points = Branch.assignAllPoints(tls_points, branches, useTrimming=True)

    for branch, expected_branch in zip(branches, expected_branches):
        assert branches_are_equal(branch, expected_branch)


def test_pre_filter_branches(base_radius, branches, removed_branch_indices):
    expected_filtered_branches = [b for idx, b in enumerate(branches) if idx not in removed_branch_indices]

    filtered_branches, pruned_points = preFilterBranches(branches, base_radius)

    filtered_branches.sort(key=lambda b: b.id)
    for branch, expected_branch in zip(filtered_branches, expected_filtered_branches):
        assert branch is expected_branch


class TestSplitCrookedBranches:
    def test_with_data(self, filtered_branches):
        split_branches = splitCrookedBranches(filtered_branches)
        for branch, expected_branch in zip(split_branches, filtered_branches):
            assert branch is expected_branch


    def test_splits_crooked_branch(self):
        skeleton_points = np.array([[0, 0, 0],
                                    [1, 0, 0],
                                    [2, 0, 0],
                                    [3, 1, 0],
                                    [3, 2, 0]], dtype=np.double)

        expected_first_half_indices = [0,1,2]
        expected_second_half_indices = [3,4]

        crooked_branch = Branch(skeleton_points)

        split_branches = splitCrookedBranches([crooked_branch])

        assert(len(split_branches) == 2)
        assert branches_are_equal(split_branches[0], Branch(skeleton_points[expected_first_half_indices]))
        assert branches_are_equal(split_branches[1], Branch(skeleton_points[expected_second_half_indices]))


    def test_does_not_split_straight_branch(self):
        skeleton_points = np.array([[0, 0, 0],
                                    [1, 0, 0],
                                    [2, 0, 0],
                                    [3, 1, 0],
                                    [4, 1, 0]], dtype=np.double)

        branch = Branch(skeleton_points)

        split_branches = splitCrookedBranches([branch])
        assert(len(split_branches) == 1)
        assert split_branches[0] is branch


def test_create_skeleton_graph(filtered_branches, parameters, branch_skeleton_points, end_point_ids, predecessors):
    skeleton_graph, skeleton_points, is_end_point = createSkeletonGraph(filtered_branches,
                                                                        parameters["minDistanceThreshold"])
    np.testing.assert_array_equal(skeleton_points, branch_skeleton_points)
    np.testing.assert_array_equal(np.flatnonzero(is_end_point), end_point_ids)

    assert(np.count_nonzero(predecessors >= 0) == len(skeleton_graph.edges))

    for edge in skeleton_graph.edges:
        assert(predecessors[edge[1]] == edge[0] or predecessors[edge[0]] == edge[1])


def test_construct_components(filtered_branches, parameters, components):
    component_stems = constructComponents(filtered_branches, parameters["minDistanceThreshold"])
    for component, expected_component in zip(component_stems, components):
        assert subtrees_are_equal(component, expected_component)
