import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra, reconstruct_path
from scipy.spatial import KDTree
from typing import Any, Dict, Tuple

from src.utils.geometry import pointSort


def generateSeeds(tlsPoints: npt.NDArray[np.double], # must be sorted by height and tlsPoints[:,2] must be height
                  parameters: Dict[str, Any]):
    tlsPoints = np.unique(tlsPoints, axis=0)
    tlsPoints = pointSort(tlsPoints, ('z','x','y'))

    basePoint, baseRadius = findBase(tlsPoints)
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


def findBase(tlsPoints: npt.NDArray[np.double]) -> Tuple[npt.NDArray[np.double], np.double]:
    bottomPoints = tlsPoints[:len(tlsPoints)//100] # points are sorted by height
    mins = np.quantile(bottomPoints, 0.05, axis=0, method="hazen")
    maxs = np.quantile(bottomPoints, 0.95, axis=0, method="hazen")

    bottomPoints = bottomPoints[np.all((bottomPoints >= mins) & (bottomPoints <= maxs), axis=-1)]

    optimum = least_squares(lambda x: (bottomPoints[:,0] - x[0])**2 + (bottomPoints[:,1] - x[1])**2 - x[2]**2,
                            np.zeros(3, dtype=np.double)).x

    basePoint = optimum.copy()
    basePoint[2] = mins[2]
    baseRadius = np.abs(optimum[2])

    return basePoint, baseRadius


def generateInitialSeeds(tlsPoints: npt.NDArray[np.double],
                         baseRadius: np.double,
                         initialSearchRange: np.double,
                         kNeighbors: int = 10,
                         basePointIndex: int = 0,
                         ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.intp], npt.NDArray[np.double]]:
    tlsPoints, basePointIndex = removeUnconnectedComponents(tlsPoints, basePointIndex, kNeighbors) # TODO: could this remove large parts of data?

    tlsKDTree = KDTree(tlsPoints)
    neighborGraph = buildNeighborGraph(tlsKDTree, kNeighbors)

    isConnected, relPathLocation, predecessors = getPathData(neighborGraph, basePointIndex)
    assert np.all(isConnected)

    isConsidered = np.zeros(len(tlsPoints), dtype=np.bool)
    isSeed = np.zeros(len(tlsPoints), dtype=np.bool)

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


def buildNeighborGraph(tlsPointsLookUp: KDTree,
                       kNeighbors: int = 10) -> csr_array:
    tlsPoints = tlsPointsLookUp.data
    neighborDistances, neighborIndices = tlsPointsLookUp.query(tlsPoints, k=kNeighbors + 1)
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
