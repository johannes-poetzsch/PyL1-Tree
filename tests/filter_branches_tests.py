import os

import numpy as np
import pytest

from src.filterBranches import filterByLength, filterTreeBranches, limitToRange, pruneThickeningLeaves
from src.objects.branch import Branch
from src.utils.iteration import breadthFirstSearch
from tests.construct_tree_graph_test import skeleton_points
from tests.global_fixtures import parameters
from tests.test_utils.comparison import subtrees_are_equal
from tests.test_utils.data_import import connect_branches, import_branches, rootPath


class TestFilterTreeBranches:
    @pytest.fixture
    def tls_points(self):
        with open(os.path.join(rootPath(), "TestData/filterBranches/tlsPoints.txt")) as f:
            return np.loadtxt(f, dtype=np.double)


    @pytest.fixture
    def input_branches_trunk(self):
        with open(os.path.join(rootPath(), "TestData/filterBranches/input/skeletonPoints.txt")) as f:
            skeleton_points = np.loadtxt(f, dtype=np.double)
        branches = import_branches("TestData/filterBranches/input", skeleton_points)
        return connect_branches(branches)[0]


    @pytest.fixture
    def filtered_branches_trunk(self, tls_points):
        with open(os.path.join(rootPath(), "TestData/filterBranches/result/skeletonPoints.txt")) as f:
            skeleton_points = np.loadtxt(f, dtype=np.double)
        branches = import_branches("TestData/filterBranches/result", skeleton_points, tls_points)
        return connect_branches(branches)[0]


    def test_on_data(self, input_branches_trunk, filtered_branches_trunk, tls_points, parameters):
        Branch.assignAllPoints(tls_points, list(breadthFirstSearch(input_branches_trunk)), useTrimming=True)
        filterTreeBranches(input_branches_trunk, parameters, pruneNewLeaves=True)
        assert subtrees_are_equal(input_branches_trunk, filtered_branches_trunk, compareTlsPoints=False)


class TestFilterByLength:
    @pytest.fixture
    def trunk(self):
        return Branch(skeletonPoints=np.array([[0,0,-6.0], [0,0,-4.7], [0,0,-3.6], [0,0,-2.4], [0,0,-1.2], [0,0,0]],
                                              dtype=np.double))

    @pytest.fixture
    def add_branch(self, trunk):
        branchCount = 0
        directions = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1]], dtype=np.double)

        def construct_branch(point_count: int, resolution: float, parent = trunk):
            nonlocal branchCount, directions
            skeleton_points = directions[branchCount] * resolution * np.arange(point_count, dtype=np.double)[:,np.newaxis]
            branch = Branch(skeleton_points, parent=parent)
            branchCount += 1
            return branch

        return construct_branch


    def test_removes_short_branches(self, trunk, add_branch):
        branches = [trunk]
        branches.append(add_branch(2, 0.5)) # totalLength: 0.5
        branches.append(add_branch(3, 0.5)) # totalLength: 1
        branches.append(add_branch(4, 1.1)) # totalLength: 3.3
        branches.append(add_branch(2, 1.3)) # totalLength: 1.3
        branches.append(add_branch(5, 0.8)) # totalLength: 3.2
        expected_remaining_branches = {trunk, branches[3], branches[5]}

        filterByLength(branches, branchSizeThreshold=4, resolution=1.5)

        remaining_branches = set(breadthFirstSearch(trunk))
        assert remaining_branches == expected_remaining_branches


    def test_considers_total_length(self, trunk, add_branch):
        branches = [trunk]
        branches.append(add_branch(3, 0.5)) # totalLength: 1
        branches.append(add_branch(3, 1.0)) # totalLength: 2
        branches.append(add_branch(3, 1.5)) # totalLength: 3
        branches.append(add_branch(3, 2.0)) # totalLength: 4
        expected_remaining_branches = {trunk, branches[3], branches[4]}

        filterByLength(branches, branchSizeThreshold=4, resolution=1.0) # min length: 3
        remaining_branches = set(breadthFirstSearch(trunk))
        assert remaining_branches == expected_remaining_branches


    def test_considers_point_count(self, trunk, add_branch):
        branches = [trunk]
        branches.append(add_branch(2, 1)) # totalLength: 1
        branches.append(add_branch(3, 1/2)) # totalLength: 1
        branches.append(add_branch(4, 1/3)) # totalLength: 1
        branches.append(add_branch(5, 1/4)) # totalLength: 1
        expected_remaining_branches = {trunk, branches[3], branches[4]}

        filterByLength(branches, branchSizeThreshold=4, resolution=1.0) # min length: 3
        remaining_branches = set(breadthFirstSearch(trunk))
        assert remaining_branches == expected_remaining_branches


    def test_returns_tls_points(self):
        tls_points = np.array([[0,0,0.3], [0,0,0.7], [0,0,1.3], [0,0,1.7]])
        removed_branch = Branch(skeletonPoints=np.array([[0,0,0], [0,0,2]], dtype=np.double),
                                tlsPoints=tls_points)
        otherBranch = Branch(skeletonPoints=np.array([[1,0,1], [1,0,2], [1,0,3]], dtype=np.double))
        branches = [removed_branch, otherBranch]

        trimmed_points = filterByLength(branches, branchSizeThreshold=3, resolution=1.5)

        assert removed_branch.isRemoved
        assert not otherBranch.isRemoved
        np.testing.assert_array_equal(trimmed_points, tls_points[[0,1]])
        np.testing.assert_array_equal(otherBranch.tlsPoints, tls_points[[2, 3]])


#def TestFilterByInfluence:



class TestLimitToRange:
    @pytest.fixture
    def branches(self):
        b1 = Branch(skeletonPoints=np.array([[0, 0, 0], [0, 0, 1]]),
                    tlsPoints=np.array([[0.2, 0, 0.5], [0, 0.4, 0.3], [0.6, 0, 0.7]]))
        b2 = Branch(skeletonPoints=np.array([[0, 0, 1], [0, 0, 2]]),
                    tlsPoints=np.array([[0.2, 0, 1.5], [0, 0.4, 1.3], [0.6, 0, 1.7]]),
                    parent=b1)
        b3 = Branch(skeletonPoints=np.array([[0, 0, 2], [0, 0, 3]]),
                    tlsPoints=np.array([[0.2, 0, 2.5], [0, 0.4, 2.3], [0.6, 0, 2.7]]),
                    parent=b2)

        return [b1, b2, b3]


    def test_limit_to_fix_range(self, branches):
        expected_remaining_points = [b.tlsPoints[[0]] for b in branches]
        expected_removed_points = np.concatenate([b.tlsPoints[1:] for b in branches], axis=0)

        removed_points = limitToRange(branches, pointRanges=0.2)

        np.testing.assert_array_equal(removed_points, expected_removed_points)
        for branch, expected_tls_points in zip(branches, expected_remaining_points):
            np.testing.assert_array_equal(branch.tlsPoints, expected_tls_points)


    def test_limit_to_range_per_branch(self, branches):
        ranges = [0.4,0.6,0.1]
        ranges = {b: r for b, r in zip(branches, ranges)}
        expected_tls_points = [branches[0].tlsPoints[:2],
                               branches[1].tlsPoints,
                               np.empty((0, 3), dtype=np.double)]
        expected_removed_points = np.concatenate((branches[0].tlsPoints[[2]], branches[2].tlsPoints))

        removed_points = limitToRange(branches, pointRanges=ranges)

        np.testing.assert_array_equal(removed_points, expected_removed_points)
        for branch, expected_points in zip(branches, expected_tls_points):
            np.testing.assert_array_equal(expected_points, branch.tlsPoints)


    def test_doubles_base_range(self, branches):
        expected_tls_points = [branches[0].tlsPoints[:2],
                               branches[1].tlsPoints[:2],
                               branches[2].tlsPoints[:1]]
        expected_removed_points = np.concatenate((branches[0].tlsPoints[2:], branches[1].tlsPoints[2:],
                                                  branches[2].tlsPoints[1:]))

        removed_points = limitToRange(branches, pointRanges=0.2, doubleBaseRange=True, baseLevelNumber=2)

        np.testing.assert_array_equal(removed_points, expected_removed_points)
        for branch, expected_points in zip(branches, expected_tls_points):
            np.testing.assert_array_equal(expected_points, branch.tlsPoints)


class TestPruneThickeningLeaves:
    @pytest.fixture
    def branch(self):
        skeletonPoints = np.array([[0,0,i] for i in range(15)], dtype=np.double)
        tlsPoints = skeletonPoints + [0.2,0,0]
        branch = Branch(skeletonPoints, tlsPoints)
        return branch


    def test_prunes_thickening_leaf(self, branch):
        original_skeleton_points = branch.skeletonPoints
        original_tls_points = branch.tlsPoints

        branch.tlsPoints[10:] += [0.3,0,0]
        removed_points = pruneThickeningLeaves([branch], branchSizeThreshold=6)

        np.testing.assert_array_equal(original_skeleton_points[:9], branch.skeletonPoints)
        np.testing.assert_array_equal(original_tls_points[:9], branch.tlsPoints)
        np.testing.assert_array_equal(original_tls_points[9:], removed_points)


    def test_only_prunes_leaves(self, branch):
        branch.tlsPoints[10:] += [0.3, 0, 0]
        original_skeleton_points = branch.skeletonPoints
        original_tls_points = branch.tlsPoints

        leaf = Branch(branch.skeletonPoints[[-1]], parent=branch)
        removed_points = pruneThickeningLeaves([branch], branchSizeThreshold=6)

        np.testing.assert_array_equal(original_skeleton_points, branch.skeletonPoints)
        np.testing.assert_array_equal(original_tls_points, branch.tlsPoints)
        assert len(removed_points) == 0


    def test_only_prunes_thickening_branches(self, branch):
        branch.tlsPoints[10:] -= [0.1,0,0]
        original_skeleton_points = branch.skeletonPoints
        original_tls_points = branch.tlsPoints

        removed_points = pruneThickeningLeaves([branch], branchSizeThreshold=6)

        np.testing.assert_array_equal(original_skeleton_points, branch.skeletonPoints)
        np.testing.assert_array_equal(original_tls_points, branch.tlsPoints)
        assert len(removed_points) == 0


    def test_only_cuts_to_size_threshold(self, branch):
        branch.tlsPoints[10:] += [0.3, 0, 0]
        original_skeleton_points = branch.skeletonPoints
        original_tls_points = branch.tlsPoints

        removed_points = pruneThickeningLeaves([branch], branchSizeThreshold=10)

        np.testing.assert_array_equal(original_skeleton_points, branch.skeletonPoints)
        np.testing.assert_array_equal(original_tls_points, branch.tlsPoints)
        assert len(removed_points) == 0


    @pytest.fixture
    def tls_points(self):
        with open(os.path.join(rootPath(), "TestData/filterBranches/pruneThickeningLeaves/tlsPoints.txt")) as f:
            return np.loadtxt(f, dtype=np.double)


    @pytest.fixture
    def input_branches(self, tls_points):
        with open(os.path.join(rootPath(), "TestData/filterBranches/pruneThickeningLeaves/input/skeletonPoints.txt")) as f:
            skeleton_points = np.loadtxt(f, dtype=np.double)
        branches = import_branches("TestData/filterBranches/pruneThickeningLeaves/input", skeleton_points, tls_points)
        return list(breadthFirstSearch(connect_branches(branches)[0]))


    @pytest.fixture
    def result_branches(self, tls_points):
        with open(os.path.join(rootPath(), "TestData/filterBranches/pruneThickeningLeaves/result/skeletonPoints.txt")) as f:
            skeleton_points = np.loadtxt(f, dtype=np.double)
        branches = import_branches("TestData/filterBranches/pruneThickeningLeaves/result", skeleton_points, tls_points)
        return list(breadthFirstSearch(connect_branches(branches)[0]))


    def test_on_data(self, input_branches, result_branches, parameters):
        pruneThickeningLeaves(input_branches, parameters["branchSizeThreshold"], useSimpleSegmentRadii=False)
        assert subtrees_are_equal(input_branches[0], result_branches[0], compareTlsPoints=True)
