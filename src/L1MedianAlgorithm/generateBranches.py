from multiprocessing import Pool
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Any, Dict, List, Tuple

from .findBranches import findNewBranches
from .runIteration import calculateDirectionalDegree, runIteration, setIterationParameters
from ..objects.branch import Branch
from ..utils.pointMetrics import calculateDensity


def generateBranches(tlsPoints: npt.NDArray[np.double],
                     seedPoints: npt.NDArray[np.double],
                     relPathLocations: npt.NDArray[np.double],
                     parameters: Dict[str, Any],
                     workers: int = 1
                     ) -> Tuple[List[Branch], npt.NDArray[np.double]]:
    skeletonBranches: List[Branch] = []
    tlsLookUp = KDTree(tlsPoints)
    skeletonData = initializeSkeleton(seedPoints, relPathLocations)
    tlsDensities = calculateDensity(None, tlsLookUp, 2 * parameters["initialSearchRange"], workers)
    searchRange = parameters["searchRange"]
    maxSearchRange = 2 * parameters["baseRadius"]

    while searchRange <= maxSearchRange:
        print(searchRange, " / ", maxSearchRange)
        for _ in range(parameters["maxIterationNumFixedRange"]):
            recomputeSkeletonPoints(skeletonData, tlsLookUp, tlsDensities, searchRange,
                                    parameters["baseRadius"], parameters["minMu"], parameters["maxMu"], workers)

        skeletonData = filterCloseDataPoints(skeletonData, parameters["nearThreshold"])
        computeSmoothDirectionalDegrees(skeletonData, parameters["kSmoothSigma"])
        setFixPoints(skeletonData, parameters["sigmaThreshold"])
        skeletonBranches.extend(findNewBranches(skeletonData,
                                                parameters["kTracingThreshold"],
                                                parameters["distanceTracingThreshold"],
                                                np.deg2rad(parameters["angleTracingThreshold"]),
                                                parameters["sigmaTracingThreshold"],
                                                parameters["branchSizeThreshold"]))

        searchRange *= 1 + parameters["iterationSpeed"]

    unfixedPoints = skeletonData["Coords"][~skeletonData["isFix"]]

    return skeletonBranches, unfixedPoints


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
                            maxMu: float,
                            workers: int = 1):
    sigma = skeletonData["Sigma"]
    changing = ~skeletonData["isFix"]

    repulsion = np.zeros((len(skeletonData), 3), dtype=np.double)
    tlsAverage = np.zeros((len(skeletonData), 3), dtype=np.double)

    setIterationParameters(baseRadius, searchRange, skeletonData, tlsDensities, tlsLookUp)
    with Pool(processes=workers) as pool:
        for idx, c, s, r, a in pool.map(runIteration, np.flatnonzero(changing), chunksize=32):
            changing[idx], sigma[idx] = c, s
            if c:
                repulsion[idx], tlsAverage[idx] = r, a

    minSigma, maxSigma = sigma.min(), sigma.max()
    sigmaRange = maxSigma - minSigma
    muRange = np.abs(maxMu - minMu)

    mu = (muRange / sigmaRange) * (sigma - minSigma) + minMu
    changing[mu < 0] = False

    skeletonData["Coords"][changing] = tlsAverage[changing] + mu[changing,np.newaxis] * repulsion[changing]


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
                                    kNeighbors: int):
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
