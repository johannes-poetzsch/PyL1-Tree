from collections import deque
import cupy as cp
import cuml.cluster as gpu_cluster
from itertools import chain, combinations
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
import sklearn.cluster as skl_cluster
from typing import Any, Dict, List, Optional, Tuple

from src.objects.branch import Branch
from src.utils.branchUtils import buildSubtreeFromGraph
from src.utils.geometry import angle, distanceAlongDirection, getValueOccurrences, normalizeVectors
from src.utils.iteration import breadthFirstSearch


def extendSkeleton(trunk: Branch,
                   unassignedTlsPoints: npt.NDArray[np.double],
                   parameters: Dict[str, Any]
                   ) -> npt.NDArray[np.double]:
    clusters = getClusters(unassignedTlsPoints,
                           radius=trunk.getDistances(trunk.tlsPoints).mean() / 2,
                           minClusterSize=parameters["branchSizeThreshold"] ** 2)

    skeletonPoints = np.concatenate([b.skeletonPoints for b in list(breadthFirstSearch(trunk))], axis=0)
    skeletonLookUp = KDTree(skeletonPoints)

    newSkeletons = []
    for cluster in clusters:
        clusterPoints = unassignedTlsPoints[cluster]
        skeleton = generateSkeleton(clusterPoints, skeletonLookUp, parameters["resolution"],
                                    np.deg2rad(parameters["minAngleThreshold"]),
                                    np.deg2rad(parameters["maxAngleThreshold"]),
                                    np.deg2rad(parameters["smoothAngleThreshold"]))
        if skeleton is not None:
            newSkeletons.append(skeleton)

    connectSkeletons(trunk, newSkeletons, parameters["resolution"], parameters["maxDistanceThreshold"])

    # could be improved by assigning clusters first
    prunedTlsPoints = Branch.reassignPoints(list(breadthFirstSearch(trunk)), useTrimming=True)

    return prunedTlsPoints


def getClusters(unassignedTlsPoints: npt.NDArray[np.double],
                radius: np.double,
                minClusterSize: int,
                keptPercentage: int = 0.99
                ) -> List[npt.NDArray[np.intp]]:
    _, clusterLabels = skl_cluster.dbscan(unassignedTlsPoints, eps=radius, min_samples=5)
    clusters = [indices for label, indices in getValueOccurrences(clusterLabels) if label >= 0]

    clusterSizes = [len(c) for c in clusters]
    maxRemovedCount = (1 - keptPercentage) * sum(clusterSizes)

    removedCount = 0
    sizes, counts = np.unique(clusterSizes, return_counts=True)
    for size, count in zip(sizes, counts):
        minClusterSize = max(minClusterSize, size)
        removedCount += count * size
        if removedCount > maxRemovedCount:
            break

    return [c for c in clusters if len(c) >= minClusterSize]


def generateSkeleton(tlsPoints: npt.NDArray[np.double], 
                     skeletonLookUp: KDTree,
                     minLength: float,
                     minAngleThreshold: float,
                     maxAngleThreshold: float,
                     smoothAngleThreshold: float,
                     testMode: bool = False) -> Optional[List[Branch]]:
    gpuTlsPoints = cp.asarray(tlsPoints)
    skeletonCandidate = None
    candidateTlsDistance = np.inf

    for skeletonSize in range(3, int(len(tlsPoints) ** 0.5) + 1):
        if testMode:
            # since randomness yields different results in python and matlab, the same randomly shuffled set of points is sampled instead.
            initPoints = tlsPoints[[(i + skeletonSize) % len(tlsPoints) for i in range(skeletonSize)]]
            skeletonPoints = (skl_cluster.KMeans(n_clusters=skeletonSize, init=initPoints, max_iter=100, tol=1e-5)
                              .fit(tlsPoints).cluster_centers_)
        else:
            if len(tlsPoints) > 1000:
                skeletonPoints = cp.asnumpy(gpu_cluster.KMeans(n_clusters=skeletonSize, max_iter=100, tol=1e-5)
                                            .fit(gpuTlsPoints).cluster_centers_)
            else:
                skeletonPoints = skl_cluster.KMeans(n_clusters=skeletonSize, max_iter=100, tol=1e-5).fit(tlsPoints).cluster_centers_
        distances, indices = skeletonLookUp.query(skeletonPoints)
        closestIdx = distances.argmin()

        rootPoint = skeletonPoints[closestIdx]
        nearestPoint = skeletonLookUp.data[indices[closestIdx]]

        initialSearchDirection = normalizeVectors(rootPoint - nearestPoint)
        rootDistances = distanceAlongDirection(rootPoint, skeletonPoints, initialSearchDirection)
        skeletonPoints = skeletonPoints[rootDistances.argsort()]

        skeletonGraph = buildSkeletonGraph(skeletonPoints, 0, initialSearchDirection, maxAngleThreshold)
        if skeletonGraph is None:
            continue

        firstBranches = [buildSubtreeFromGraph(skeletonGraph, skeletonPoints, startIdx=0, nextIdx=s)
                         for s in skeletonGraph.successors(0)]
        
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


def connectSkeletons(trunk: Branch,
                     skeletons: List[List[Branch]],
                     resolution: float,
                     maxDistanceThreshold: float):
    connections: Dict[Branch, Tuple[Optional[Branch], int, float]] = {s[0]: (None, 0, np.inf) for s in skeletons}
    newStartBranch = trunk

    while skeletons:
        newBranches = [b for b in breadthFirstSearch(newStartBranch) if b is not trunk]
        for branch in newBranches:
            branch.interpolate(resolution, temporary=True)
        skeletonPoints = np.concatenate([b.skeletonPoints for b in newBranches], axis=0)
        indices = np.concatenate([np.arange(len(b), dtype=np.intp) for b in newBranches], axis=0)
        branchIndices = np.repeat(np.arange(len(newBranches), dtype=np.intp), [len(b) for b in newBranches])
        skeletonLookUp = KDTree(skeletonPoints)

        for skeleton in skeletons:
            distance = connections[skeleton[0]][-1]
            newDistance, idx = skeletonLookUp.query(skeleton[0][0])
            if newDistance < distance:
                connections[skeleton[0]] = (newBranches[branchIndices[idx]], indices[idx], newDistance)

        skeleton = min(skeletons, key=lambda s: connections[s[0]][-1])
        del skeletons[skeletons.index(skeleton)]

        junctionBranch, junctionIdx, distance = connections[skeleton[0]]
        if distance > maxDistanceThreshold:
            break

        junctionBranch, splitEnd = junctionBranch.split(junctionIdx)
        if splitEnd is not None:
            for s in skeletons:
                b, i, d = connections[s[0]]
                if b is junctionBranch and i > junctionIdx:
                    connections[s[0]] = (splitEnd, i-junctionIdx, d)

        newStartBranch = Branch(np.stack([junctionBranch[-1], skeleton[0][0]]), parent=junctionBranch)
        for startBranch in skeleton:
            startBranch.setParent(newStartBranch)

    for branch in reversed(list(breadthFirstSearch(trunk))):
        branch.removeTemporaryPoints()
        if len(branch.children) == 1:
            branch.expand()


def buildSkeletonGraph(skeletonPoints: npt.NDArray[np.double],
                       rootIdx: int,
                       initialSearchDirection: npt.NDArray[np.double],
                       maxAngleThreshold: float) -> Optional[nx.DiGraph]:
    kdTree = KDTree(skeletonPoints)
        
    neighbors = []
    for pointIdx, point in enumerate(skeletonPoints):
        minNeighborDistance = kdTree.query(point, k=2)[0][1]
        pointNeighbors = kdTree.query_ball_point(point, r=2*minNeighborDistance)
        neighbors.append(pointNeighbors)

    skeletonGraph = nx.DiGraph()
    skeletonGraph.add_nodes_from(range(len(skeletonPoints)))

    searchHooks = deque([(rootIdx, initialSearchDirection)])
    while searchHooks:
        pointIdx, searchDirection = searchHooks.popleft()

        for neighborIdx in neighbors[pointIdx]:
            if (neighborIdx == pointIdx or neighborIdx == rootIdx or # root has no incoming edges
                    skeletonGraph.has_edge(pointIdx, neighborIdx)):
                continue

            neighborDirection = normalizeVectors(skeletonPoints[neighborIdx] - skeletonPoints[pointIdx])
            neighborAngle = angle(searchDirection, neighborDirection)

            if neighborAngle <= maxAngleThreshold:
                distance = np.linalg.norm(skeletonPoints[neighborIdx] - skeletonPoints[pointIdx], axis=-1)
                skeletonGraph.add_edge(pointIdx, neighborIdx, distance=distance)
                searchHooks.append((neighborIdx, neighborDirection))

    if not nx.is_weakly_connected(skeletonGraph):
        return None

    skeletonGraph: nx.DiGraph = nx.minimum_spanning_arborescence(skeletonGraph, attr="distance")
    return skeletonGraph


def skeletonIsValid(startBranches: List[Branch],
                    minLength: float,
                    minAngleThreshold: float,
                    smoothAngleThreshold: float) -> bool:
    branches = list(chain.from_iterable([breadthFirstSearch(b) for b in startBranches]))
    if any([b.length < minLength for b in branches]):
        return False

    for [b, s] in chain(combinations(startBranches, 2),
                        chain.from_iterable([combinations(b.children, 2) for b in branches])):
        branchDirection = normalizeVectors(b.skeletonPoints[-1] - b.skeletonPoints[0])
        siblingDirection = normalizeVectors(s.skeletonPoints[-1] - s.skeletonPoints[0])
        if angle(branchDirection, siblingDirection) < minAngleThreshold:
            return False
        
    for b in branches:
        if np.any(b.getAngles() > smoothAngleThreshold):
            return False
        
    return True
        
