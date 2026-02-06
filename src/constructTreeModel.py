from itertools import chain, combinations
import networkx as nx
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List

from scipy.spatial import KDTree

from connectComponents import connectComponents
from filterBranches import filterUncenteredBranches, filterByInfluence
from objects.branch import Branch
from utils.branchUtils import buildSubtreeFromGraph, condenseSkeletonPointIds, splitAndGetEndPointIds
from utils.iteration import breadthFirstSearch


def constructTreeModel(tlsPoints: npt.NDArray[np.double],
                       branches: List[Branch],
                       parameters: Dict[str, Any],
                       heightAttribute: int = 2):
    Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)
    branches = filterBranches(branches, parameters["baseRadius"])

    skeletonPointCount = condenseSkeletonPointIds(branches)
    skeletonPoints = np.empty([skeletonPointCount, 3], dtype=np.double)
    for b in branches:
        skeletonPoints[b.skeletonIDs] = b.skeletonPoints

    endPointIds = list(chain.from_iterable([splitAndGetEndPointIds(b) for b in branches]))
    isEndPoint = np.zeros(len(skeletonPoints), dtype=bool)
    isEndPoint[endPointIds] = True

    componentStems = constructComponents(branches, skeletonPoints, endPointIds,
                                         parameters["minDistanceThreshold"], heightAttribute)
    rootComponentStem = min(componentStems, key=lambda componentStem: componentStem.skeletonPoints[0][heightAttribute])

    branches = list(chain.from_iterable([breadthFirstSearch(stem) for stem in componentStems]))
    unassignedPoints = Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)

    stem, prunedPoints = connectComponents(rootComponentStem, componentStems, unassignedPoints, isEndPoint,
                                           parameters["minDistanceThreshold"], parameters["maxDistanceThreshold"],
                                           np.deg2rad(parameters["minAngleThreshold"]),
                                           np.deg2rad(parameters["maxAngleThreshold"]),
                                           parameters["resolution"], parameters["windowSize"],
                                           parameters["densityFactor"])


def filterBranches(branches: List[Branch],
                   baseRadius: float) -> List[Branch]:
    filterUncenteredBranches(branches)
    branches = [b for b in branches if not b.isRemoved]
    # TODO: Step 1.2.2 does not work in original implementation -> implement anyways or keep behavior?

    branches.sort(key=lambda b: b.length, reverse=True)

    branchTlsDistances = [b.getDistances(b.tlsPoints) for b in branches]
    searchRanges = np.array([np.quantile(d, 0.9, method="hazen") for d in branchTlsDistances])
    filterByInfluence(branches, searchRanges=np.minimum(searchRanges, baseRadius))

    return [b for b in branches if not b.isRemoved]


def constructComponents(branches: List[Branch],
                        skeletonPoints: npt.NDArray[np.double],
                        endPointIds: List[int],
                        minDistanceThreshold: float,
                        heightAttribute: int = 2) -> List[Branch]:
    componentStems = []

    skeletonGraph = createSkeletonGraph(branches, skeletonPoints, endPointIds, minDistanceThreshold)
    for c in nx.connected_components(skeletonGraph):
        if len(c) <= 1:
            raise ValueError("skeletonGraph should not contain single-point components")
        # find the terminal node with the lowest height
        terminalNodes = [node for node in c if skeletonGraph.degree(node) == 1]
        rootNode = min(terminalNodes, key=lambda node: skeletonPoints[node][heightAttribute])

        componentSkeletonGraph = nx.dfs_tree(skeletonGraph, rootNode)
        componentStem = buildSubtreeFromGraph(componentSkeletonGraph, skeletonPoints, rootNode)
        componentStems.append(componentStem)

    return componentStems


def createSkeletonGraph(branches: List[Branch],
                        skeletonPoints: npt.NDArray[np.double],
                        endPointIds: List[int],
                        minDistanceThreshold: float) -> nx.Graph:
    skeletonGraph = nx.Graph()
    skeletonGraph.add_nodes_from(range(len(skeletonPoints)))

    for b in branches:
        pointPairs = [(b.skeletonIDs[i], b.skeletonIDs[i+1], np.linalg.norm(b.skeletonPoints[i+1] - b.skeletonPoints[i]))
                      for i in range(len(b.skeletonPoints) - 1)]
        skeletonGraph.add_weighted_edges_from(pointPairs, weight="distance")

    endPointsLookUp = KDTree(skeletonPoints[endPointIds])
    endPointNeighbors = endPointsLookUp.query_ball_tree(endPointsLookUp, minDistanceThreshold)
    for index1, neighbors in enumerate(endPointNeighbors):
        id1 = endPointIds[index1]
        for index2 in neighbors:
            id2 = endPointIds[index2]
            if id1 >= id2:
                continue
            distance = np.linalg.norm(skeletonPoints[id1] - skeletonPoints[id2])
            skeletonGraph.add_edge(id1, id2, distance=distance)

    skeletonGraph = nx.minimum_spanning_tree(skeletonGraph, weight="distance")

    return skeletonGraph
