from itertools import chain
import time
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
                       unfixedPoints: npt.NDArray[np.double],
                       parameters: Dict[str, Any],
                       heightAttribute: int = 2,
                       analytics: Dict[str, Any] | None = None,
                       workers: int = 1,
                       useMultiprocessing: bool = True
                       ) -> Tuple[Branch, npt.NDArray[np.double]]:
    Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)
    branches, _ = preFilterBranches(branches, parameters["baseRadius"])
    branches = splitCrookedBranches(branches)

    componentStems = constructComponents(branches, parameters["minDistanceThreshold"], heightAttribute)
    trunk = min(componentStems, key=lambda componentStem: componentStem.skeletonPoints[0, heightAttribute])

    branches = list(chain.from_iterable([breadthFirstSearch(stem) for stem in componentStems]))
    trimmedPoints = Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)

    connectionStart = time.perf_counter()
    prunedBranches, trimmedPoints = connectComponents(
        trunk, componentStems, unfixedPoints, trimmedPoints,
        parameters["minDistanceThreshold"], parameters["maxDistanceThreshold"],
        np.deg2rad(parameters["minAngleThreshold"]), np.deg2rad(parameters["maxAngleThreshold"]),
        parameters["resolution"], parameters["windowSize"], parameters["densityFactor"],
        workers=workers, useMultiprocessing=useMultiprocessing)
    if analytics is not None:
        analytics["time_component_connection_s"] = time.perf_counter() - connectionStart

    trimmedPoints = addBasePoint(trunk, parameters["basePoint"], trimmedPoints)
    filteringStart = time.perf_counter()
    prunedPoints1 = filterTreeBranches(trunk, parameters, pruneNewLeaves=True)
    if analytics is not None:
        analytics["time_initial_tree_branch_filtering_s"] = time.perf_counter() - filteringStart
    branches = list(breadthFirstSearch(trunk))

    prunedPoints2 = removeTlsPointsNearPoints(branches, unfixedPoints, parameters["resolution"])

    prunedPoints3 = limitToRange(branches, parameters["baseRadius"], doubleBaseRange=True, baseLevelNumber=3)
    prunedPoints4 = pruneThickeningLeaves(branches, parameters["branchSizeThreshold"])

    prunedPoints = ([prunedPoints1, prunedPoints2, prunedPoints3, prunedPoints4, trimmedPoints]
                    + [b.tlsPoints for b in prunedBranches])
    prunedPoints = np.concatenate(prunedPoints, axis=0)

    return trunk, prunedPoints


def addBasePoint(trunk: Branch, basePoint: npt.NDArray[np.double], unassignedTlsPoints: npt.NDArray[np.double]):
    rootPoint = trunk.skeletonPoints[0]
    stemDirection = normalizeVectors(trunk.skeletonPoints[-1] - rootPoint)
    baseDirection = normalizeVectors(rootPoint - basePoint)
    baseAngle = angle(stemDirection, baseDirection)

    if np.rad2deg(baseAngle) < 90:
        trunk.skeletonPoints = np.insert(trunk.skeletonPoints, 0, basePoint, axis=0)
        trunk.updateSkeletonMeasurements()
        unassignedTlsPoints = Branch.assignLocalPoints(unassignedTlsPoints, list(breadthFirstSearch(trunk)),
                                                       [trunk], useTrimming=True)

    return unassignedTlsPoints


def createSkeletonGraph(branches: List[Branch],
                        minDistanceThreshold: float
                        ) -> Tuple[nx.Graph, npt.NDArray[np.double], npt.NDArray[bool]]:
    skeletonPoints = np.concatenate([b.skeletonPoints for b in branches], axis=0)

    isEndPoint = np.zeros(len(skeletonPoints), dtype=bool)
    branchStartIdx = 0
    for branch in branches:
        isEndPoint[[branchStartIdx, branchStartIdx + len(branch) - 1]] = True
        branchStartIdx += len(branch)
    endPointIds = np.flatnonzero(isEndPoint)

    skeletonGraph = nx.Graph()
    skeletonGraph.add_nodes_from(range(len(skeletonPoints)))

    offset = 0
    for b in branches:
        pointPairs = [(offset + i, offset + i+1, np.linalg.norm(b.skeletonPoints[i+1] - b.skeletonPoints[i]))
                      for i in range(len(b) - 1)]
        skeletonGraph.add_weighted_edges_from(pointPairs, weight="distance")
        offset += len(b)

    endPointsLookUp = KDTree(skeletonPoints[endPointIds])
    endPointNeighbors = endPointsLookUp.query_ball_tree(endPointsLookUp, minDistanceThreshold)
    for index1, neighbors in enumerate(endPointNeighbors):
        id1 = endPointIds[index1]
        for index2 in neighbors:
            id2 = endPointIds[index2]
            distance = np.linalg.norm(skeletonPoints[id1] - skeletonPoints[id2])
            skeletonGraph.add_edge(id1, id2, distance=distance)

    skeletonGraph = nx.minimum_spanning_tree(skeletonGraph, weight="distance")

    return skeletonGraph, skeletonPoints, isEndPoint


def constructComponents(branches: List[Branch],
                        minDistanceThreshold: float,
                        heightAttribute: int = 2) -> List[Branch]:
    componentStems = []

    skeletonGraph, skeletonPoints, isEndPoint = createSkeletonGraph(branches, minDistanceThreshold)
    for c in nx.connected_components(skeletonGraph):
        if len(c) <= 1:
            raise ValueError("skeletonGraph should not contain single-point components")
        # find the terminal node with the lowest height
        terminalNodes = [node for node in c if skeletonGraph.degree(node) == 1]
        rootNode = min(terminalNodes, key=lambda node: skeletonPoints[node, heightAttribute])

        componentSkeletonGraph = nx.dfs_tree(skeletonGraph, rootNode)
        componentStem = buildSubtreeFromGraph(componentSkeletonGraph, skeletonPoints, rootNode, isEndPoint=isEndPoint)
        componentStems.append(componentStem)

    return componentStems


def preFilterBranches(branches: List[Branch],
                      baseRadius: float
                      ) -> Tuple[List[Branch], npt.NDArray[np.double]]:
    pruneUncenteredBranches(branches)
    removedBranches = [b for b in branches if b.isRemoved]
    branches = [b for b in branches if not b.isRemoved]
    # TODO: Step 1.2.2 does not work in original implementation -> implement anyways or keep behavior?

    branches.sort(key=lambda b: b.length, reverse=True)

    branchTlsDistances = [b.getDistances(b.tlsPoints) for b in branches]
    searchRanges = [np.minimum(baseRadius, np.quantile(d, 0.9, method="hazen")) if len(d) > 0 else 0.0
                    for d in branchTlsDistances]
    searchRanges = {b: np.minimum(baseRadius, r) for b, r in zip(branches, searchRanges)}
    filterByInfluence(branches, searchRanges, useMean=True)

    removedBranches.extend([b for b in branches if b.isRemoved])
    branches = [b for b in branches if not b.isRemoved]

    prunedPoints = np.concatenate([b.tlsPoints for b in removedBranches], axis=0) \
        if removedBranches else np.empty((0,3), dtype=np.double)

    return branches, prunedPoints


def removeTlsPointsNearPoints(branches: List[Branch],
                              points: npt.NDArray[np.double],
                              resolution: float
                              ) -> npt.NDArray[np.double]:
    unassignedPoints = [np.empty((0,3), dtype=np.double)]
    branchIndices, distancesToBranches = Branch.findGlobalClosestBranches(points, branches)
    points = points[distancesToBranches > resolution]
    distancesToBranches = distancesToBranches[distancesToBranches > resolution]

    isOutside = [b.getDistances(point[np.newaxis], trimStart=b.isStem(), trimEnd=b.isLeaf(), returnTrimming=True)[1].squeeze()
                 for point, b in zip(points, [branches[idx] for idx in branchIndices])]
    isOutside = np.array(isOutside, dtype=np.bool)

    for branch in branches:
        if len(branch.tlsPoints) == 0:
            continue
        tlsPoints = branch.tlsPoints
        distances = branch.getDistances(tlsPoints)
        radius = np.quantile(distances, 0.9, method="hazen")
        relevantPoints = points[isOutside | (distancesToBranches > radius)]

        distancesToPoints, _ = KDTree(relevantPoints).query(tlsPoints)

        unassignedPoints.append(tlsPoints[distances >= distancesToPoints])
        branch.tlsPoints = tlsPoints[distances < distancesToPoints]

    unassignedPoints = np.concatenate(unassignedPoints, axis=0)
    return unassignedPoints


def splitCrookedBranches(branches: List[Branch]):
    straightBranches = []
    for branch in branches:
        if len(branch) < 4:
            straightBranches.append(branch)
            continue

        startDirection = normalizeVectors(branch.skeletonPoints[1] - branch.skeletonPoints[0])
        endDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[-2])

        if angle(startDirection, endDirection) < np.pi / 2:  # acute angles are fine
            straightBranches.append(branch)
            continue

        firstPartDirections = normalizeVectors(branch.skeletonPoints[1:-2] - branch.skeletonPoints[0])
        secondPartDirections = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[2:-1])

        splitAngles = angle(firstPartDirections, secondPartDirections)
        splitIdx = np.argmax(splitAngles) + 1

        splitBranches = [Branch(branch.skeletonPoints[:splitIdx+1]), Branch(branch.skeletonPoints[splitIdx+1:])]
        straightBranches.extend(splitBranches)

    return straightBranches
