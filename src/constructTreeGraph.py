from itertools import chain
import networkx as nx
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Tuple

from scipy.spatial import KDTree

from src.connectComponents import connectComponents
from src.filterBranches import (filterTreeBranches, filterByInfluence, limitToRange,
                                pruneThickeningLeaves, pruneUncenteredBranches)
from src.objects.branch import Branch
from src.utils.geometry import angle, normalizeVectors
from src.utils.branchUtils import buildSubtreeFromGraph
from src.utils.iteration import breadthFirstSearch


def constructTreeGraph(tlsPoints: npt.NDArray[np.double],
                       branches: List[Branch],
                       parameters: Dict[str, Any],
                       heightAttribute: int = 2
                       ) -> Tuple[Branch, npt.NDArray[np.double]]:
    Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)
    branches, _ = preFilterBranches(branches, parameters["baseRadius"])

    componentStems = constructComponents(branches, parameters["minDistanceThreshold"], heightAttribute)
    rootComponentStem = min(componentStems, key=lambda componentStem: componentStem.skeletonPoints[0][heightAttribute])

    branches = list(chain.from_iterable([breadthFirstSearch(stem) for stem in componentStems]))
    trimmedPoints = Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)

    stem, prunedBranches, trimmedPoints = (
        connectComponents(rootComponentStem, componentStems, trimmedPoints,
                          parameters["minDistanceThreshold"], parameters["maxDistanceThreshold"],
                          np.deg2rad(parameters["minAngleThreshold"]),
                          np.deg2rad(parameters["maxAngleThreshold"]),
                          parameters["resolution"], parameters["windowSize"],
                          parameters["densityFactor"]))

    trimmedPoints = addBasePoint(stem, parameters["basePoint"], trimmedPoints)
    prunedPoints1 = filterTreeBranches(stem, parameters, pruneNewLeaves=True)
    branches = list(breadthFirstSearch(stem))
    prunedPoints2 = limitToRange(branches, parameters["baseRadius"], doubleBaseRange=True, baseLevelNumber=3)
    prunedPoints3 = pruneThickeningLeaves(branches, parameters["branchSizeThreshold"])

    prunedPoints = [prunedPoints1, prunedPoints2, prunedPoints3, trimmedPoints] + [b.tlsPoints for b in prunedBranches]
    prunedPoints = np.concatenate(prunedPoints, axis=0)

    return stem, prunedPoints


def addBasePoint(stem: Branch, basePoint: npt.NDArray[np.double], unassignedTlsPoints: npt.NDArray[np.double]):
    rootPoint = stem.skeletonPoints[0]
    stemDirection = normalizeVectors(stem.skeletonPoints[-1] - rootPoint)
    baseDirection = normalizeVectors(rootPoint - basePoint)
    baseAngle = angle(stemDirection, baseDirection)

    if np.rad2deg(baseAngle) < 90:
        stem.skeletonPoints = np.insert(stem.skeletonPoints, 0, basePoint, axis=0)
        stem.updateSkeletonMeasurements()
        unassignedTlsPoints = Branch.assignLocalPoints(unassignedTlsPoints, [stem], useTrimming=True)

    return unassignedTlsPoints


def createSkeletonGraph(branches: List[Branch],
                        skeletonPoints: npt.NDArray[np.double],
                        isEndPoint: npt.NDArray[np.bool],
                        minDistanceThreshold: float) -> nx.Graph:
    skeletonGraph = nx.Graph()
    skeletonGraph.add_nodes_from(range(len(skeletonPoints)))

    offset = 0
    for b in branches:
        pointPairs = [(offset + i, offset + i+1, np.linalg.norm(b.skeletonPoints[i+1] - b.skeletonPoints[i]))
                      for i in range(len(b.skeletonPoints) - 1)]
        skeletonGraph.add_weighted_edges_from(pointPairs, weight="distance")
        offset += len(b.skeletonPoints)

    endPointIds = np.flatnonzero(isEndPoint)
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


def constructComponents(branches: List[Branch],
                        minDistanceThreshold: float,
                        heightAttribute: int = 2) -> List[Branch]:
    componentStems = []

    skeletonPoints = np.concatenate([b.skeletonPoints for b in branches], axis=0)

    isEndPoint = np.empty(len(skeletonPoints), dtype=bool)
    offset = 0
    for branch in branches:
        pointCount = len(branch.skeletonPoints)
        isEndPoint[offset:offset+pointCount] = branch.isEndPoint(np.arange(pointCount, dtype=int))
        offset += pointCount

    skeletonGraph = createSkeletonGraph(branches, skeletonPoints, isEndPoint, minDistanceThreshold)
    for c in nx.connected_components(skeletonGraph):
        if len(c) <= 1:
            raise ValueError("skeletonGraph should not contain single-point components")
        # find the terminal node with the lowest height
        terminalNodes = [node for node in c if skeletonGraph.degree(node) == 1]
        rootNode = min(terminalNodes, key=lambda node: skeletonPoints[node][heightAttribute])

        componentSkeletonGraph = nx.dfs_tree(skeletonGraph, rootNode)
        componentStem = buildSubtreeFromGraph(componentSkeletonGraph, skeletonPoints, rootNode, isEndPoint=isEndPoint)
        componentStems.append(componentStem)

    return componentStems


def preFilterBranches(branches: List[Branch],
                      baseRadius: float
                      ) -> Tuple[List[Branch], npt.NDArray[np.double]]:
    pruneUncenteredBranches(branches)
    branches = [b for b in branches if not b.isRemoved]
    removedBranches = [b for b in branches if b.isRemoved]
    # TODO: Step 1.2.2 does not work in original implementation -> implement anyways or keep behavior?

    branches.sort(key=lambda b: b.length, reverse=True)

    branchTlsDistances = [b.getDistances(b.tlsPoints) for b in branches]

    searchRanges = np.array([np.quantile(d, 0.9, method="hazen") if len(d) > 0 else 0.0
                             for d in branchTlsDistances])
    filterByInfluence(branches, searchRanges=np.minimum(searchRanges, baseRadius))

    branches = [b for b in branches if not b.isRemoved]
    removedBranches.extend([b for b in branches if b.isRemoved])

    prunedPoints = np.concatenate([b.tlsPoints for b in branches], axis=0)

    return branches, prunedPoints
