from collections import deque
from itertools import chain, combinations
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from sklearn.cluster import dbscan, k_means
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional

from .objects.branch import Branch
from .utils.geometry import angle, normalizeVectors
from .utils.iteration import breadthFirstSearch


def extendSkeleton(stem: Branch, 
                   unassignedTlsPoints: npt.NDArray[np.double],
                   parameters: Dict[str, any]
                   ) -> npt.NDArray[np.double]:
    clusters = getClusters(unassignedTlsPoints, 
                           radius=stem.getDistances(stem.tlsPoints).mean() / 2,
                           minSize=parameters["branchSizeThreshold"] ** 2)
    
    skeletonLookUp, treeSkeletonPoints = getSkeletonPointLookUp(stem)
    (iSkeletonLookUp, iTreeSkeletonPoints, branches, 
     iSkeletonBranchIndices, interpolationIndices) = getInterpolatedSkeletonLookUp(stem, parameters["resolution"])

    newSubtreeRoots = []
    for cluster in clusters:
        clusterPoints = unassignedTlsPoints[cluster]
        skeleton = generateSkeleton(clusterPoints, skeletonLookUp, treeSkeletonPoints, 
                                    parameters["resolution"], 
                                    np.deg2rad(parameters["minAngleThreshold"]),
                                    np.deg2rad(parameters["maxAngleThreshold"]),
                                    np.deg2rad(parameters["smoothAngleThreshold"]))

        junctionIdx = iSkeletonLookUp.query(skeleton[0].skeletonPoints[0])
        connectionBranch = connectSkeleton(skeleton,
                                           junctionPoint=iTreeSkeletonPoints[junctionIdx],
                                           junctionBranch=branches[iSkeletonBranchIndices[junctionIdx]],
                                           junctionIdx=interpolationIndices[junctionIdx],
                                           isInterpolation=interpolationIndices[junctionIdx] == interpolationIndices[junctionIdx - 1])
        newSubtreeRoots.append(connectionBranch)

    prunedTlsPoints = []
    branches = list(breadthFirstSearch(stem)) # TODO: make this an extra method?

    for cluster, root in zip(clusters, newSubtreeRoots):
        prunedTlsPoints.append(Branch.reassignPoints(unassignedTlsPoints[cluster],
                                                     branches,
                                                     closeBranches=list(breadthFirstSearch(root)),
                                                     useTrimming=True))

    for branch in branches:
        tlsPoints = branch.tlsPoints
        branch.tlsPoints = np.empty([0,3], dtype=np.double)
        prunedTlsPoints.append(Branch.reassignPoints(tlsPoints, 
                                                     branches, 
                                                     closeBranches=[branch], 
                                                     useTrimming=True))

    return prunedTlsPoints


def getClusters(unassignedTlsPoints: npt.NDArray[np.double],
                radius: np.double,
                minSize: int,
                prunePercentage = 0.01
                ) -> List[npt.NDArray[np.intp]]:
    _, clusterLabels = dbscan(unassignedTlsPoints, eps=radius, min_samples=5)
    clusterNum = clusterLabels.max() + 1

    clusters = []
    for idx in range(clusterNum):
        clusters.append(np.flatnonzero(clusterLabels == idx))

    clusters.sort(key=lambda c: len(c))
    totalSize = sum([len(c) for c in clusters])

    removedCount = 0
    while True:
        nextRemovedCount = removedCount
        nextSize = len(clusters[nextRemovedCount])
        while len(clusters[nextRemovedCount]) == nextSize and nextRemovedCount+1 < len(clusters):
            nextRemovedCount += 1
        nextSum = sum([len(c) for c in clusters[:idx+1]])
        if nextSize >= minSize and nextSum / totalSize > prunePercentage:
            break
        else:
            removedCount = nextRemovedCount

    clusters = clusters[removedCount:]

    return clusters


def getSkeletonPointLookUp(stem: Branch):
    branches = list(breadthFirstSearch(stem))
    treeSkeletonPoints = [stem.skeletonPoints] + [b.skeletonPoints[1:] for b in branches[1:]]
    treeSkeletonPoints = np.concatenate(treeSkeletonPoints, axis=0)

    skeletonLookUp = KDTree(treeSkeletonPoints)

    return skeletonLookUp, treeSkeletonPoints


def getInterpolatedSkeletonLookUp(stem: Branch, resolution: float):
    branches = list(breadthFirstSearch(stem))
    interpolations = [b.getInterpolation(resolution) for b in branches]
    stemPoints, stemIndices = interpolations[0]

    interpolationIndices = [stemIndices] + [indices[1:] for _, indices in interpolations[1:]]
    interpolationIndices = np.concatenate(interpolationIndices, axis=0)

    treeSkeletonPoints = [stemPoints] + [points[1:] for points, _ in interpolations[1:]]
    skeletonBranchIndices = np.concatenate([np.full(len(points), idx) for idx, points in enumerate(treeSkeletonPoints)])
    treeSkeletonPoints = np.concatenate(treeSkeletonPoints, axis=0)

    skeletonLookUp = KDTree(treeSkeletonPoints)

    return skeletonLookUp, treeSkeletonPoints, branches, skeletonBranchIndices, interpolationIndices


def generateSkeleton(tlsPoints: npt.NDArray[np.double], 
                     skeletonLookUp: KDTree,
                     treeSkeletonPoints: npt.NDArray[np.double],
                     minLength: float,
                     minAngleThreshold: float,
                     maxAngleThreshold: float,
                     smoothAngleThreshold: float) -> Optional[List[Branch]]:
    skeletonCandidate = None
    candidateTlsDistance = np.inf

    for skeletonSize in range(3, int(len(tlsPoints) ** 0.5) + 1):
        skeletonPoints, _, _, _ = k_means(tlsPoints, skeletonSize)
        distances, indices = skeletonLookUp.query(skeletonPoints)
        closestIdx = distances.argmin()

        rootPoint = skeletonPoints[closestIdx]
        nearestPoint = treeSkeletonPoints[indices[closestIdx]]

        initialSearchDirection = normalizeVectors(rootPoint - nearestPoint)
        treeDistances: npt.NDArray = np.vecdot(skeletonPoints - rootPoint, initialSearchDirection)
        skeletonPoints = skeletonPoints[treeDistances.argsort()]

        skeletonGraph = buildSkeletonGraph(skeletonPoints, 0, initialSearchDirection, maxAngleThreshold)
        if skeletonGraph is None:
            continue

        firstBranches = [buildSubtreeFromGraph(skeletonGraph, skeletonPoints, 0, s) for s in skeletonGraph.successors(0)]
        
        if not skeletonIsValid(firstBranches, minLength, minAngleThreshold, smoothAngleThreshold):
            continue

        tlsDistances = np.full(len(tlsPoints), np.inf, dtype=np.double)
        for b in chain.from_iterable([breadthFirstSearch(b) for b in firstBranches]):
            tlsDistances = np.minimum(tlsDistances, b.getDistances(tlsPoints))
        
        newTlsDistance = tlsDistances.mean()

        if newTlsDistance < candidateTlsDistance:
            skeletonCandidate = firstBranches
            candidateTlsDistance = newTlsDistance
    
    return skeletonCandidate


def connectSkeleton(skeleton: List[Branch],
                    junctionPoint: npt.NDArray[np.double],
                    junctionBranch: Branch,
                    junctionIdx: int,
                    isInterpolation: bool
                    ) -> Branch:
    if isInterpolation:
        junctionIdx += 1
        junctionBranch.skeletonPoints = np.insert(junctionBranch.skeletonPoints, junctionIdx, junctionPoint, axis=0)
        junctionBranch.updateSkeletonMeasurements()

    junctionBranch = junctionBranch.split(junctionIdx)

    rootPoint = skeleton[0].skeletonPoints[0]
    connectionBranch = Branch(id=-1, 
                              skeletonIDs=np.arange(2), 
                              skeletonPoints=np.stack([junctionPoint, rootPoint]), 
                              parent=junctionBranch)
    
    for branch in skeleton:
        branch.setParent(connectionBranch)
    if len(skeleton) == 1:
        connectionBranch.expand()

    return connectionBranch


def buildSkeletonGraph(skeletonPoints: npt.NDArray[np.double],
                       rootIdx: int,
                       initialSearchDirection: npt.NDArray[np.double],
                       maxAngleThreshold: float) -> Optional[nx.DiGraph]:
    kdTree = KDTree(skeletonPoints)
        
    neighbors = []
    for point in skeletonPoints:
        minNeighborDistance = kdTree.query(point, k=2)[0][1]
        pointNeighbors = kdTree.query_ball_point(point, r=2*minNeighborDistance, return_sorted=True)
        neighbors.append(pointNeighbors[1:])

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(skeletonPoints)))

    searchHooks = deque([(rootIdx, initialSearchDirection)])
    while searchHooks:
        pointIdx, searchDirection = searchHooks.popleft()

        for neighborIdx in neighbors[pointIdx]:
            if (neighborIdx == rootIdx # root has no incoming edges
                or graph.has_edge(pointIdx, neighborIdx)):
                continue

            neighborDirection = normalizeVectors(skeletonPoints[neighborIdx] - skeletonPoints[pointIdx])
            neighborAngle = angle(searchDirection, neighborDirection)

            if neighborAngle <= maxAngleThreshold:
                distance = np.linalg.norm(skeletonPoints[neighborIdx] - skeletonPoints[pointIdx])
                graph.add_edge(pointIdx, neighborIdx, distance=distance)
                searchHooks.append((neighborIdx, neighborDirection))

    if nx.is_weakly_connected(skeletonGraph):
        return None

    skeletonGraph: nx.DiGraph = nx.minimum_spanning_arborescence(skeletonGraph, attr="distance")

    return graph


def buildSubtreeFromGraph(skeletonGraph: nx.DiGraph,
                          skeletonPoints: npt.NDArray[np.double],
                          rootIdx: int,
                          nextIdx: int,
                          parent: Branch = None):
    pointIndices = [rootIdx, nextIdx]

    while True:
        successors = list(skeletonGraph.successors(pointIndices[-1]))
        if len(successors) == 1:
            pointIndices.append(successors[0])
        else:
            break
    
    branch = Branch(id=-1,
                    skeletonIDs=np.arange(len(pointIndices)),
                    skeletonPoints=skeletonPoints[pointIndices],
                    parent=parent)
    
    for s in successors:
        buildSubtreeFromGraph(skeletonGraph, skeletonPoints, pointIndices[-1], s, branch)

    return branch


def skeletonIsValid(firstBranches: List[Branch],
                    minLength: float,
                    minAngleThreshold: float,
                    smoothAngleThreshold: float) -> bool:
    branches = chain.from_iterable([breadthFirstSearch(b) for b in firstBranches])
    if sum([b.length for b in branches]) < minLength:
        return False

    for [b, s] in chain.from_iterable([combinations(firstBranches, 2), 
                                       chain.from_iterable([combinations(b.children, 2) for b in branches])]):
        branchDirection = normalizeVectors(b.skeletonPoints[-1] - b.skeletonPoints[0])
        siblingDirection = normalizeVectors(s.skeletonPoints[-1] - s.skeletonPoints[0])
        if angle(branchDirection, siblingDirection) < minAngleThreshold:
            return False
        
    for b in branches:
        if np.any(b.getAngles() > smoothAngleThreshold):
            return False
        
    return True
        
