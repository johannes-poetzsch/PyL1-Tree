import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Any, Dict, List, Optional

from .findBranches import findNewBranches
from ..objects.branch import Branch
from ..utils.pointMetrics import calculateDensity, decayingWeights


def generateBranches(tlsPoints: npt.NDArray[np.double],
                     seedPoints: npt.NDArray[np.double],
                     relPathLocations: npt.NDArray[np.double],
                     parameters: Dict[str, Any]) -> List[Branch]:
    skeletonBranches: List[Branch] = []
    tlsLookUp = KDTree(tlsPoints)
    skeletonData = initializeSkeleton(seedPoints, relPathLocations)
    tlsDensities = calculateDensity(None, tlsLookUp, 2 * parameters["initialSearchRange"])

    while parameters["searchRange"] <= 2 * parameters["baseRadius"]:
        print(parameters["searchRange"], " / ", 2 * parameters["baseRadius"])
        for _ in range(parameters["maxIterationNumFixedRange"]):
            recomputeSkeletonPoints(skeletonData, tlsLookUp, tlsDensities,
                                    parameters["searchRange"], parameters["baseRadius"],
                                    parameters["minMu"], parameters["maxMu"])

        skeletonData = filterCloseDataPoints(skeletonData, parameters["nearThreshold"])
        computeSmoothDirectionalDegrees(skeletonData, parameters["kSmoothSigma"])
        setFixPoints(skeletonData, parameters["sigmaThreshold"])
        skeletonBranches.extend(findNewBranches(skeletonData,
                                                parameters["kTracingThreshold"],
                                                parameters["distanceTracingThreshold"],
                                                np.deg2rad(parameters["angleTracingThreshold"]),
                                                parameters["sigmaTracingThreshold"],
                                                parameters["branchSizeThreshold"]))

        parameters["searchRange"] *= 1 + parameters["iterationSpeed"]

    return skeletonBranches


def initializeSkeleton(seedPoints: npt.NDArray[np.double],
                       relPathLocations: npt.NDArray[np.double]) -> npt.NDArray:
    skeletonData = np.zeros(len(seedPoints),
                            dtype=[("Coords", (np.double, 3)),
                                   ("relPathLocation", np.double),
                                   ("Sigma", np.double),
                                   ("isFix", np.bool_),
                                   ("isBranch", np.bool_),
                                   ])
    skeletonData["Coords"] = seedPoints
    skeletonData["relPathLocation"] = relPathLocations

    return skeletonData


def recomputeSkeletonPoints(skeletonData: npt.NDArray,
                            tlsLookUp: KDTree,
                            tlsDensities: npt.NDArray[np.double],
                            searchRange: float,
                            baseRadius: float,
                            minMu: float,
                            maxMu: float) -> npt.NDArray[np.double]:
    coords = skeletonData["Coords"]
    skeletonLookUp = KDTree(coords)
    notFix = np.logical_not(skeletonData["isFix"])
    notFixCount = np.count_nonzero(notFix)
    toNotFixIdx = np.empty(len(skeletonData), dtype=np.intp)
    toNotFixIdx[notFix] = np.arange(notFixCount)
    changing = np.ones(notFixCount, dtype=bool)

    neighborRadii = np.maximum(2 * searchRange, skeletonData["relPathLocation"][notFix] * 2 * baseRadius)
    neighborLists = skeletonLookUp.query_ball_point(coords[notFix], neighborRadii)

    repulsion = np.zeros([notFixCount, 3], dtype=np.double)
    for idx, dataPoint in enumerate(skeletonData):
        if dataPoint["isFix"]:
            continue

        nfIdx = toNotFixIdx[idx]
        neighbors = neighborLists[nfIdx]
        if len(neighbors) <= 1:
            changing[nfIdx] = False
            dataPoint["Sigma"] = 0.5
            continue

        neighbors = [n for n in neighbors if n != idx]

        neighborVectors = dataPoint["Coords"] - coords[neighbors]
        neighborDistances = np.linalg.norm(neighborVectors, axis=-1)

        dataPoint["Sigma"] = calculateDirectionalDegree(neighborVectors, neighborDistances, searchRange)
        repulsion[nfIdx] = calculateRepulsion(neighborVectors, neighborDistances, searchRange)

    tlsNeighborLists = tlsLookUp.query_ball_point(coords[notFix], neighborRadii)
    tlsAverage = np.zeros([notFixCount, 3], dtype=np.double)
    for idx, point in enumerate(coords[notFix]):
        tlsNeighbors = tlsNeighborLists[idx]
        if len(tlsNeighbors) == 0:
            changing[idx] = False
            continue

        neighborDensities = tlsDensities[tlsNeighbors]
        tlsNeighborPoints = tlsLookUp.data[tlsNeighbors]

        tlsAverage[idx] = calculatePointAverage(point, tlsNeighborPoints, neighborDensities, searchRange)

    minSigma, maxSigma = skeletonData["Sigma"].min(), skeletonData["Sigma"].max()
    sigmaRange = maxSigma - minSigma
    muRange = np.abs(maxMu - minMu)

    mu = (muRange / sigmaRange) * (skeletonData["Sigma"][notFix] - minSigma) + minMu
    changing[mu < 0] = False

    coords[np.flatnonzero(notFix)[changing]] = tlsAverage[changing] + mu[changing][:,np.newaxis] * repulsion[changing]


def calculateDirectionalDegree(neighborVectors: npt.NDArray[np.double],
                               neighborDistances: Optional[npt.NDArray[np.double]] = None,
                               searchRange: Optional[float] = None,
                               useDecayingWeights: bool = True) -> np.double:
    if len(neighborVectors) <= 3:
        return 0.5

    if useDecayingWeights:
        assert(searchRange is not None and neighborDistances is not None)
        weights = decayingWeights(neighborDistances, 2. * searchRange)
    else:
        weights = np.ones(1, dtype=np.double)
    c = np.matmul(np.matrix_transpose(neighborVectors * weights[:,np.newaxis]), neighborVectors)
    eigenValues, _ = np.linalg.eig(c)

    sigma = eigenValues.max(axis=-1) / eigenValues.sum(axis=-1)
    return sigma


def calculateRepulsion(neighborVectors: npt.NDArray[np.double],
                       neighborDistances: npt.NDArray[np.double],
                       searchRange: float
                       ) -> npt.NDArray[np.double]:
    neighborDistances = np.clip(neighborDistances, a_min=0.001 * searchRange, a_max=None)

    weights = decayingWeights(neighborDistances, 2. * searchRange)
    weights /= neighborDistances

    repulsionTerm = weights.transpose() @ neighborVectors

    return repulsionTerm / weights.sum()


def calculatePointAverage(point: npt.NDArray[np.double],
                          neighborPoints: npt.NDArray[np.double],
                          densities: npt.NDArray[np.double],
                          searchRange: float
                          ) -> npt.NDArray[np.double]:
    distances = np.linalg.norm(neighborPoints - point, axis=-1)
    np.clip(distances, a_min=0.001 * searchRange, a_max=None, out=distances)

    weights = decayingWeights(distances, 2 * searchRange)
    weights *= densities

    averageTerm = weights.transpose() @ neighborPoints

    return averageTerm / weights.sum()


def filterCloseDataPoints(skeletonData: npt.NDArray,
                          nearThreshold: float
                          ) -> npt.NDArray:
    kept = np.ones_like(skeletonData, dtype=bool)

    skeletonLookUp = KDTree(skeletonData["Coords"])
    # TODO: individual queries might be faster since some can be skipped
    skeletonNeighbors = skeletonLookUp.query_ball_tree(skeletonLookUp, nearThreshold)
    for idx in range(len(skeletonData)):
        if not kept[idx] or len(skeletonNeighbors[idx]) <= 1:
            continue

        neighbors = np.array(skeletonNeighbors[idx])
        changing = ~skeletonData["isFix"][neighbors]
        kept[neighbors[changing]] = False
        kept[idx] = True

    return skeletonData[kept]


def computeSmoothDirectionalDegrees(skeletonData: npt.NDArray,
                                    kNeighbors: int
                                    ) -> npt.NDArray:
    skeletonLookUp = KDTree(skeletonData["Coords"])
    notFix = ~skeletonData["isFix"] # (n)
    notFixIndices = np.flatnonzero(notFix)

    _, neighbors = skeletonLookUp.query(skeletonData[notFix]["Coords"], kNeighbors + 1) # (nF,k+1)
    neighbors = neighbors[:,1:] # (nF,k)

    if np.any(neighbors == len(skeletonData)):
        raise ValueError("Found too few neighbors to calculate smooth directional degrees")

    neighborIsValid = notFix[neighbors] # (nF,k)
    validNeighborsCount = np.count_nonzero(neighborIsValid, axis=1) # (nF)

    hasEnoughValidNeighbors = validNeighborsCount > 3 # (nF)

    for idx in range(len(notFixIndices)):
        if not hasEnoughValidNeighbors[idx]:
            skeletonData["Sigma"][notFixIndices[idx]] = 0.95
            continue

        originalIdx = notFixIndices[idx]
        neighborVectors = (skeletonData["Coords"][neighbors[idx][neighborIsValid[idx]]]
                           - skeletonData["Coords"][originalIdx]) # (nF,3)
        skeletonData["Sigma"][originalIdx] = calculateDirectionalDegree(neighborVectors, useDecayingWeights=False)

    invSigma = 1.0 - skeletonData["Sigma"] # (n)

    neighborInvSigmaSum = np.where(neighborIsValid, invSigma[neighbors], 0.0).sum(axis=-1) # (nF)
    sigma = 1.0 - ((invSigma[notFix] + neighborInvSigmaSum) / (1.0 + validNeighborsCount))

    skeletonData["Sigma"][notFix] = np.where(sigma < 0, 0.5, sigma)


def setFixPoints(skeletonData: npt.NDArray,
                 sigmaThreshold: float):
    isBranch = skeletonData["isBranch"]
    skeletonData["isFix"][~isBranch] = skeletonData["Sigma"][~isBranch] > sigmaThreshold
