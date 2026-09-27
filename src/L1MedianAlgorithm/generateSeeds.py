from heapq import heappop, heappush

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra, reconstruct_path
from scipy.spatial import KDTree
from typing import Any, Dict, List, Optional, Tuple

from src.removeStump import findBaseWithoutStump
from src.objects.boundedObject import BoundedObject
from src.utils.geometry import getValueOccurrences, pointSort


def generateSeeds(tlsPoints: npt.NDArray[np.double], # must be sorted by height and tlsPoints[:,2] must be height
                  parameters: Dict[str, Any],
                  useStumpRemoval: bool = False,
                  useRansac: bool = False):
    tlsPoints = np.unique(tlsPoints, axis=0)
    tlsPoints = pointSort(tlsPoints, ('z','x','y'))

    basePoint, baseRadius, tlsPoints = findBase(tlsPoints, useStumpRemoval, parameters["resolution"], useRansac)
    basePointIndex = len(tlsPoints)
    tlsPoints = np.append(tlsPoints, basePoint[np.newaxis], axis=0)

    parameters["basePoint"] = basePoint
    if parameters["baseRadius"] is None: parameters["baseRadius"] = baseRadius
    if parameters["initialSearchRange"] is None: parameters["initialSearchRange"] = baseRadius * 0.1
    if parameters["searchRange"] is None: parameters["searchRange"] = baseRadius * 0.1

    tlsPoints, seedIndices, relPathPositions = generateInitialSeeds(tlsPoints,
                                                                    parameters["baseRadius"],
                                                                    parameters["initialSearchRange"],
                                                                    basePointIndex=basePointIndex)

    seedIndices = filterSeeds(tlsPoints, seedIndices, parameters["nearThreshold"])
    seedPoints = tlsPoints[seedIndices]
    seedRelPathPositions = relPathPositions[seedIndices]

    return seedPoints, seedRelPathPositions, tlsPoints


def findBase(tlsPoints: npt.NDArray[np.double], # sorted by height
             removeStump: bool = False,
             resolution: float = 0.3,
             useRansac: bool = False
             ) -> Tuple[npt.NDArray[np.double], float, npt.NDArray[np.double]]:
    if removeStump:
        basePoint, baseRadius, tlsPoints = findBaseWithoutStump(tlsPoints, resolution, useRansac=useRansac)
        if baseRadius is not None:
            return basePoint, baseRadius, tlsPoints

    bottomPoints = tlsPoints[:len(tlsPoints)//100]
    mins = np.quantile(bottomPoints, 0.05, axis=0, method="hazen")
    maxs = np.quantile(bottomPoints, 0.95, axis=0, method="hazen")
    points = bottomPoints[np.all((bottomPoints >= mins) & (bottomPoints <= maxs), axis=-1)]
    optimum = least_squares(lambda x: (points[:, 0] - x[0]) ** 2 + (points[:, 1] - x[1]) ** 2 - x[2] ** 2,
                            np.zeros(3, dtype=np.double)).x
    baseRadius = np.abs(optimum[2])
    basePoint = optimum.copy()
    basePoint[2] = mins[2]

    return basePoint, baseRadius, tlsPoints


def generateInitialSeeds(tlsPoints: npt.NDArray[np.double],
                         baseRadius: np.double,
                         initialSearchRange: np.double,
                         kNeighbors: int = 10,
                         basePointIndex: int = 0,
                         ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.intp], npt.NDArray[np.double]]:
    #tlsPoints, basePointIndex = removeUnconnectedComponents(tlsPoints, basePointIndex, kNeighbors)
    #neighborGraph = buildNeighborGraph(tlsKDTree, kNeighbors)
    neighborGraph = buildConnectedGraph(tlsPoints, basePointIndex, kNeighbors)

    isConnected, relPathLocation, predecessors = getPathData(neighborGraph, basePointIndex)
    tlsPoints, relPathLocation = tlsPoints[isConnected], relPathLocation[isConnected]

    idxShift = np.empty(len(predecessors), dtype=np.intp)
    idxShift[isConnected] = np.arange(len(tlsPoints))
    predecessors = idxShift[predecessors[isConnected]]
    basePointIndex = idxShift[basePointIndex]

    isConsidered = np.zeros(len(tlsPoints), dtype=np.bool)
    isSeed = np.zeros(len(tlsPoints), dtype=np.bool)
    tlsKDTree = KDTree(tlsPoints)

    for seed in np.argsort(relPathLocation, kind="stable"):
        if isConsidered[seed]:
            continue

        while not isSeed[seed]:
            isSeed[seed] = True

            searchRange = max(2 * baseRadius * np.sqrt(relPathLocation[seed]), 2 * initialSearchRange)
            neighborIndices = tlsKDTree.query_ball_point(tlsPoints[seed], searchRange)

            isConsidered[neighborIndices] = True

            if seed == basePointIndex:
                break
            seed = predecessors[seed]

    seedIndices = np.flatnonzero(isSeed)
    return tlsPoints, seedIndices, relPathLocation


def filterSeeds(tlsPoints: npt.NDArray[np.double],
                seedIndices: npt.NDArray[np.intp],
                nearThreshold: float) -> npt.NDArray[np.intp]:
    seedPoints = tlsPoints[seedIndices]
    seedKDTree = KDTree(seedPoints)

    isSeed = np.ones_like(seedIndices, dtype=np.bool)

    for idx in range(len(seedIndices)):
        if isSeed[idx]:
            neighbors = seedKDTree.query_ball_point(seedPoints[idx], 2 * nearThreshold)
            isSeed[neighbors] = False
            isSeed[idx] = True

    return seedIndices[isSeed]


def buildNeighborGraph(tlsLookUp: KDTree,
                       kNeighbors: int = 10) -> csr_array:
    tlsPoints = tlsLookUp.data
    neighborDistances, neighborIndices = tlsLookUp.query(tlsPoints, k=kNeighbors + 1)
    neighborDistances, neighborIndices = neighborDistances[:, 1:], neighborIndices[:, 1:]

    pointCount = len(tlsPoints)
    neighborGraph = csr_array((neighborDistances.flatten(),
                               neighborIndices.flatten(),
                               np.arange(pointCount + 1) * kNeighbors),
                              shape=(pointCount, pointCount))
    return neighborGraph


def removeUnconnectedComponents(tlsPoints: npt.NDArray[np.double],
                                basePointIndex: int,
                                kNeighbors: int = 10) -> Tuple[npt.NDArray[np.double], int]:
    tlsKDTree = KDTree(tlsPoints)
    neighborGraph = buildNeighborGraph(tlsKDTree, kNeighbors)
    _, componentLabels = connected_components(neighborGraph, directed=False)

    isConnected = componentLabels == componentLabels[basePointIndex]
    tlsPoints = tlsPoints[isConnected]

    newBasePointIdx = np.flatnonzero(np.flatnonzero(isConnected) == basePointIndex)[0]

    return tlsPoints, newBasePointIdx


def buildConnectedGraph(tlsPoints: npt.NDArray[np.double],
                        basePointIndex: int,
                        kNeighbors: int = 10,
                        maxOutlierRate: float = 0.01,
                        minComponentSize: int = 100) -> csr_array:
    pointCount = len(tlsPoints)
    tlsKDTree = KDTree(tlsPoints)
    neighborGraph = buildNeighborGraph(tlsKDTree, kNeighbors)
    componentCount, componentLabels = connected_components(neighborGraph, directed=False)

    components = []
    for label, indices in getValueOccurrences(componentLabels):
        components.append(indices)
    componentBoundingBoxes = [(tlsPoints[c].min(axis=0), tlsPoints[c].max(axis=0)) for c in components]
    componentLookUps: List[Optional[KDTree]] = componentCount * [None]

    connectedPointCount = 0
    isConnected = np.zeros(componentCount, dtype=np.bool)
    isTooSmall = np.array([len(c) < minComponentSize for c in components], dtype=np.bool)
    connections = componentCount * [(np.inf, -1, -1, -1)] # distance, start component, start idx, end idx
    closeComponentPairs = []

    baseComponentIdx = componentLabels[basePointIndex]
    lastConnectedIdx = baseComponentIdx

    while True:
        isConnected[lastConnectedIdx] = True
        connectedPointCount += len(components[lastConnectedIdx])
        if np.all(isConnected | isTooSmall) or connectedPointCount > (1 - maxOutlierRate) * pointCount:
            break

        for c in range(componentCount):
            if not (isConnected[c] or isTooSmall[c]):
                boundingBoxDistance = BoundedObject.getBBDistance(componentBoundingBoxes[lastConnectedIdx],
                                                                  componentBoundingBoxes[c])
                heappush(closeComponentPairs, (boundingBoxDistance, lastConnectedIdx, c))

        minDistance = min([c[0] for c in connections])
        while len(closeComponentPairs) > 0 and minDistance > closeComponentPairs[0][0]:
            bbDistance, connectorIdx, componentIdx = heappop(closeComponentPairs)
            if isConnected[componentIdx] or isTooSmall[componentIdx]:
                continue
            if componentLookUps[connectorIdx] is None:
                componentLookUps[connectorIdx] = KDTree(tlsPoints[components[connectorIdx]])

            lookUp: KDTree = componentLookUps[connectorIdx]
            distances, indices = lookUp.query(tlsPoints[components[componentIdx]])
            closest = np.argmin(distances)
            distance = distances[closest]
            if distance < connections[componentIdx][0]:
                connections[componentIdx] = (distance, connectorIdx, indices[closest], closest)
                minDistance = min(distance, minDistance)

        unconnectedComponents = [c for c in range(componentCount) if not isConnected[c]]
        newConnectedComponent = min(unconnectedComponents, key=lambda c: connections[c][0])
        lastConnectedIdx = newConnectedComponent

    # insert all new connections into the csr-array at once
    rows = []
    cols = []
    distances = []
    for componentIdx in range(componentCount):
        if not isConnected[componentIdx] or componentIdx == baseComponentIdx:
            continue
        distance, connectorIdx, startIdx, endIdx = connections[componentIdx]
        # the connections only store the point index within the components, not the global point index
        startIdx = components[connectorIdx][startIdx]
        endIdx = components[componentIdx][endIdx]
        rows.extend([startIdx, endIdx])
        cols.extend([endIdx, startIdx])
        distances.append(distance)
    neighborGraph[np.array(rows), np.array(cols)] = np.repeat(distances, 2)

    return neighborGraph


def getPathData(neighborGraph: csr_array,
                basePointIndex: int = 0
                ) -> Tuple[npt.NDArray[bool], npt.NDArray[np.double], npt.NDArray[np.intp]]:
    _, predecessors = dijkstra(neighborGraph, indices=basePointIndex, directed=False, return_predecessors=True)
    predecessors = predecessors.flatten()
    pathTree = reconstruct_path(neighborGraph, predecessors, directed=False)

    isConnected = predecessors >= 0
    isConnected[basePointIndex] = True

    bfsOrder: npt.NDArray = breadth_first_order(pathTree, basePointIndex, directed=False, return_predecessors=False)

    pathIndex = np.zeros_like(predecessors, dtype=int)
    for idx in bfsOrder[1:]:
        pathIndex[idx] = pathIndex[predecessors[idx]] + 1

    pathLength = pathIndex.copy() + 1
    for idx in np.flip(bfsOrder)[:-1]:
        pathLength[predecessors[idx]] = np.maximum(pathLength[predecessors[idx]], pathLength[idx])

    pathIndex = pathLength - pathIndex
    relPathLocation = pathIndex / pathLength

    return isConnected, relPathLocation, predecessors
