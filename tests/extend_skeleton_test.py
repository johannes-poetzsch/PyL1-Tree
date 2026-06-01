import os

import numpy as np
import pytest
from scipy.spatial import KDTree

from src.extendSkeleton import connectSkeletons, generateSkeleton, getClusters
from src.objects.branch import Branch
from src.utils.geometry import pointSort
from src.utils.iteration import breadthFirstSearch
from tests.global_fixtures import parameters
from tests.test_utils.comparison import subtrees_are_equal
from tests.test_utils.data_import import connect_branches, import_branches, rootPath


@pytest.fixture
def tls_points():
    with open(os.path.join(rootPath(), "TestData/adjustBifurcations/tlsPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


@pytest.fixture
def input_branches_trunk(tls_points):
    with open(os.path.join(rootPath(), "TestData/smoothBranches/smoothBranch/result/skeletonPoints.txt")) as f:
        skeleton_points = np.loadtxt(f, dtype=np.double)
    branches = import_branches("TestData/extendSkeleton/input", skeleton_points, tls_points)
    return connect_branches(branches)[0]


@pytest.fixture
def skeleton_points(input_branches_trunk):
    return np.concatenate([b.skeletonPoints for b in list(breadthFirstSearch(input_branches_trunk))], axis=0)


@pytest.fixture
def trunk_radius(input_branches_trunk):
    return input_branches_trunk.getDistances(input_branches_trunk.tlsPoints).mean() / 2


@pytest.fixture
def expected_clusters():
    clusters = []
    for d in os.listdir(os.path.join(rootPath(), "TestData/extendSkeleton/skeletons")):
        with open(os.path.join(rootPath(), "TestData/extendSkeleton/skeletons", d, "cluster.txt")) as f:
            clusters.append(np.loadtxt(f, dtype=np.double))
    return clusters


@pytest.fixture
def new_skeletons():
    skeletons = []
    for d in os.listdir(os.path.join(rootPath(), "TestData/extendSkeleton/skeletons")):
        path = os.path.join(rootPath(), "TestData/extendSkeleton/skeletons", d)
        if os.path.isfile(os.path.join(path, "branches.txt")):
            with open(os.path.join(path, "skeletonPoints.txt")) as f:
                skeleton_points = np.loadtxt(f, dtype=np.double)
            branches = import_branches(path, skeleton_points)
            skeletons.append(connect_branches(branches))
    return skeletons


@pytest.fixture
def result_branches_trunk():
    with open(os.path.join(rootPath(), "TestData/extendSkeleton/result/skeletonPoints.txt")) as f:
        skeleton_points = np.loadtxt(f, dtype=np.double)
    branches = import_branches("TestData/extendSkeleton/result", skeleton_points)
    return connect_branches(branches)[0]


@pytest.fixture
def unassigned_points(tls_points):
    with open(os.path.join(rootPath(), "TestData/extendSkeleton/input/closestBranches.txt")) as f:
        closestBranches = np.loadtxt(f, dtype=np.intp)
    return tls_points[closestBranches == 0]


def test_get_clusters(trunk_radius, unassigned_points, parameters, expected_clusters):
    clusters = getClusters(unassigned_points, trunk_radius, parameters["branchSizeThreshold"] ** 2)

    assert len(expected_clusters) == len(clusters)
    clusters = [unassigned_points[c] for c in clusters]

    for c, ec in zip(clusters, expected_clusters):
        np.testing.assert_array_equal(pointSort(c), pointSort(ec))


def test_generate_skeleton(unassigned_points, expected_clusters, skeleton_points, parameters, new_skeletons):
    skeletons = []
    skeleton_lookup = KDTree(skeleton_points)

    for c in expected_clusters:
        skeleton = generateSkeleton(c, skeleton_lookup, parameters["resolution"],
                                    np.deg2rad(parameters["minAngleThreshold"]), np.deg2rad(parameters["maxAngleThreshold"]),
                                    np.deg2rad(parameters["smoothAngleThreshold"]), testMode=True)
        if skeleton is not None:
            skeletons.append(skeleton)

    assert len(skeletons) == len(new_skeletons)
    for skeleton, expected_skeleton in zip(skeletons, new_skeletons):
        # since the skeletons may have multiple first branches,
        # these are added to a dummy parent to connect them to a single subtree
        stem = Branch(skeleton[0][[0,0]])
        expected_stem = Branch(expected_skeleton[0][[0,0]])

        for b in skeleton:
            b.setParent(stem)
        for b in expected_skeleton:
            b.setParent(expected_stem)

        assert subtrees_are_equal(stem, expected_stem, compareTlsPoints=False, atol=1e-6) # original rounds to 6 decimals


def test_connect_skeletons(input_branches_trunk, new_skeletons, parameters, result_branches_trunk):
    connectSkeletons(input_branches_trunk, new_skeletons, parameters["resolution"], parameters["maxDistanceThreshold"])
    assert subtrees_are_equal(result_branches_trunk, input_branches_trunk, compareTlsPoints=False)
