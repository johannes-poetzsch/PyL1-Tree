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
                       unfixedPoints: npt.NDArray[np.double],
                       parameters: Dict[str, Any],
                       heightAttribute: int = 2
                       ) -> Tuple[Branch, npt.NDArray[np.double]]:
    Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)
    branches, _ = preFilterBranches(branches, parameters["baseRadius"])
    branches = splitCrookedBranches(branches)

    componentStems = constructComponents(branches, parameters["minDistanceThreshold"], heightAttribute)
    stem = min(componentStems, key=lambda componentStem: componentStem.skeletonPoints[0, heightAttribute])

    branches = list(chain.from_iterable([breadthFirstSearch(stem) for stem in componentStems]))
    trimmedPoints = Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)

    prunedBranches, trimmedPoints = connectComponents(
        stem, componentStems, unfixedPoints, trimmedPoints,
        parameters["minDistanceThreshold"], parameters["maxDistanceThreshold"],
        np.deg2rad(parameters["minAngleThreshold"]), np.deg2rad(parameters["maxAngleThreshold"]),
        parameters["resolution"], parameters["windowSize"], parameters["densityFactor"])

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

    searchRanges = np.array([np.quantile(d, 0.9, method="hazen") if len(d) > 0 else 0.0
                             for d in branchTlsDistances])
    filterByInfluence(branches, searchRanges=np.minimum(searchRanges, baseRadius), useMean=True)

    removedBranches.extend([b for b in branches if b.isRemoved])
    branches = [b for b in branches if not b.isRemoved]

    prunedPoints = np.concatenate([b.tlsPoints for b in removedBranches], axis=0) \
        if removedBranches else np.empty((0,3), dtype=np.double)

    return branches, prunedPoints


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
