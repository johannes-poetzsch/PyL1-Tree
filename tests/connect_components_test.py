import itertools
from itertools import chain, combinations
import numpy as np
import os
import pytest
from scipy.spatial import KDTree

from src.connectComponents import connectComponents, findConnections, generatePotentialConnections
from src.objects.branch import Branch
from src.objects.component import ConnectedComponent, UnconnectedComponent
from src.utils.iteration import breadthFirstSearch
from tests.global_fixtures import base_point, base_radius, parameters, tls_points
from tests.utils.comparison import branches_are_equal, subtrees_are_equal
from tests.utils.data_import import import_branches, rootPath


@pytest.fixture
def branches(skeleton_points, end_point_ids, tls_points):
    return import_branches("TestData/constructTreeGraph/components", skeleton_points,
                           end_point_ids=end_point_ids)


@pytest.fixture
def components_stems(branches):
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

    component_stems.sort(key=lambda c: min([b.skeletonPoints[:,2].min() for b in breadthFirstSearch(c)]))
    return component_stems


@pytest.fixture(scope="module")
def connection_strategies(skeleton_points):
    connection_strategies = {}
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/connectionStrategies/componentIds.txt")) as f:
        component_indices = np.loadtxt(f, dtype=int) - 1
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/connectionStrategies/lengths.txt")) as f:
        lengths = np.loadtxt(f, dtype=np.double)
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/connectionStrategies/meanDistances.txt")) as f:
        mean_distances = np.loadtxt(f, dtype=np.double)
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/connectionStrategies/tlsPointCounts.txt")) as f:
        tls_point_counts = np.loadtxt(f, dtype=int)

    scores = tls_point_counts / lengths / mean_distances

    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/connectionStrategies/pointIds.txt")) as f:
        for line, score, component_idx in zip(f, scores, component_indices):
            point_ids = np.fromstring(line, dtype=int, sep=" ") - 1
            connection_strip = skeleton_points[point_ids]

            if component_idx not in connection_strategies:
                connection_strategies[component_idx] = []
            connection_strategies[component_idx].append((score, connection_strip))

    for strategies in connection_strategies.values():
        strategies.sort(key=lambda s: s[0], reverse=True)

    return connection_strategies


@pytest.fixture(scope='module')
def end_point_ids():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/endPointIds.txt")) as f:
        return np.loadtxt(f, dtype=int) - 1


@pytest.fixture(scope='module')
def potential_connection_edges(skeleton_points):
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/potentialConnections/edges.txt")) as f:
        edges = np.loadtxt(f, dtype=int) - 1
    edges = skeleton_points[edges]
    return edges


@pytest.fixture(scope='module')
def is_trimmed():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/components/isTrimmed.txt")) as f:
        return np.loadtxt(f, dtype=bool)


@pytest.fixture
def potential_connections(branches, unfixed_points, parameters):
    unfixedLookUp, unfixedWrapper = KDTree(unfixed_points), Branch(unfixed_points)
    potentialConnections = generatePotentialConnections(branches, unfixedLookUp, unfixedWrapper,
                                                        parameters["minDistanceThreshold"])
    return potentialConnections


@pytest.fixture(scope='module')
def skeleton_points(unfixed_points):
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/branchSkeletonPoints.txt")) as f:
        branch_skeleton_points = np.loadtxt(f, dtype=np.double)
    return np.append(branch_skeleton_points, unfixed_points, axis=0)


@pytest.fixture(scope='module')
def assigned_points(is_trimmed, tls_points):
    return tls_points[~is_trimmed]


@pytest.fixture
def assigned_distances(assigned_points, branches):
    distances = Branch.findMinBranchDistances(assigned_points, branches)
    return distances


@pytest.fixture(scope='module')
def unassigned_points(is_trimmed, tls_points):
    return tls_points[is_trimmed]


@pytest.fixture
def unassigned_distances(unassigned_points, branches):
    distances = Branch.findMinBranchDistances(unassigned_points, branches)
    return distances


@pytest.fixture(scope='module')
def unfixed_points():
    with open(os.path.join(rootPath(), "TestData/constructTreeGraph/unfixedPoints.txt")) as f:
        return np.loadtxt(f, dtype=np.double)


def test_generate_potential_connections(potential_connection_edges, branches, unfixed_points, parameters):
    unfixedLookUp, unfixedWrapper = KDTree(unfixed_points), Branch(unfixed_points)
    potentialConnections = generatePotentialConnections(branches, unfixedLookUp, unfixedWrapper,
                                                        parameters["minDistanceThreshold"])
    edges = [np.stack((u[0][u[1]], v[0][v[1]])) for u,v in potentialConnections.edges]

    # sort edge points so that the edge orientation does not matter
    edges, expected_edges = [np.sort(e, axis=0) for e in edges], [np.sort(e, axis=0) for e in potential_connection_edges]

    edges, expected_edges = np.stack(edges).sort(axis=0), np.stack(expected_edges).sort(axis=0)

    # expected edges might appear multiple times in edges
    # since points can appear in multiple branches and a new (duplicate) edge is created for each branch
    np.testing.assert_array_equal(np.unique(edges), expected_edges)


@pytest.mark.parametrize("connection_type", ["simple connections", "indirect connections"])
def test_find_connections(connection_type, branches, components_stems, connection_strategies, parameters, potential_connections,
                          assigned_points, assigned_distances, unassigned_points, unassigned_distances):
    if connection_type == "simple connections":
        component_idx = 1
    elif connection_type == "indirect connections":
        component_idx = 51

    assigned_points_look_up = KDTree(assigned_points)
    Branch.assignAllPoints(assigned_points, branches)

    root_component = ConnectedComponent(list(breadthFirstSearch(components_stems[0])), assigned_points_look_up, assigned_distances)
    other_component = UnconnectedComponent(list(breadthFirstSearch(components_stems[component_idx])))

    findConnections(root_component, other_component, potential_connections,
                    KDTree(unassigned_points), unassigned_distances,
                    parameters["minDistanceThreshold"], parameters["maxDistanceThreshold"],
                    np.deg2rad(parameters["minAngleThreshold"]), np.deg2rad(parameters["maxAngleThreshold"]),
                    parameters["windowSize"], parameters["densityFactor"])

    connections = other_component.connections

    expected_connections = connection_strategies[component_idx]
    assert len(connections) == len(expected_connections)
    for connection, expected_connection in zip(connections, expected_connections):
        connection_strip = connection[5]
        expected_connection_strip \
            = np.flipud(expected_connection[-1])
        np.testing.assert_array_equal(connection_strip, expected_connection_strip)

        score = connection[0]
        expected_score = expected_connection[0]
        assert score == pytest.approx(expected_score)


def test_connect_components(branches, components_stems, parameters, unfixed_points, assigned_points, unassigned_points):
    Branch.assignAllPoints(assigned_points, branches)

    trunk = components_stems[0]

    connectComponents(trunk, components_stems, unfixed_points, unassigned_points,
                      parameters["minDistanceThreshold"], parameters["maxDistanceThreshold"],
                      np.deg2rad(parameters["minAngleThreshold"]), np.deg2rad(parameters["maxAngleThreshold"]),
                      parameters["resolution"], parameters["windowSize"], parameters["densityFactor"])

    assert True


