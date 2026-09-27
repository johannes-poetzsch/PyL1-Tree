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
        #print(searchRange, " / ", maxSearchRange)
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
    changingIndices = np.flatnonzero(changing)

    setIterationParameters(baseRadius, searchRange, skeletonData, tlsDensities, tlsLookUp)

    if workers > 1:
        with Pool(processes=workers) as pool:
            results = pool.map(runIteration, changingIndices, chunksize=32)
    else:
        results = map(runIteration, changingIndices)

    repulsion = np.zeros((len(changingIndices), 3), dtype=np.double)
    tlsAverage = np.zeros((len(changingIndices), 3), dtype=np.double)
    updated = np.zeros(len(changingIndices), dtype=np.bool_)

    for resultIndex, (idx, canChange, pointSigma, pointRepulsion, pointAverage) in enumerate(results):
        changing[idx] = canChange
        sigma[idx] = pointSigma
        if canChange:
            updated[resultIndex] = True
            repulsion[resultIndex] = pointRepulsion
            tlsAverage[resultIndex] = pointAverage

    minSigma = sigma.min()
    maxSigma = sigma.max()
    sigmaRange = maxSigma - minSigma

    muRange = np.abs(maxMu - minMu)
    mu = (muRange / sigmaRange) * (sigma - minSigma) + minMu

    changing[mu < 0] = False
    compactChanging = updated & changing[changingIndices]
    updatedIndices = changingIndices[compactChanging]

    skeletonData["Coords"][updatedIndices] = (
            tlsAverage[compactChanging]
            + mu[updatedIndices, np.newaxis]
            * repulsion[compactChanging]
    )


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

    if len(notFixIndices) == 0:
        return

    _, neighbors = skeletonLookUp.query(skeletonData[notFix]["Coords"], kNeighbors + 1) # (nF,k+1)
    neighbors = neighbors[:,1:] # (nF,k)

    if np.any(neighbors == len(skeletonData)):
        raise ValueError("Found too few neighbors to calculate smooth directional degrees")

    neighborIsValid = notFix[neighbors] # (nF,k)
    validNeighborsCount = np.count_nonzero(neighborIsValid, axis=1) # (nF)

    hasEnoughValidNeighbors = validNeighborsCount > 3 # (nF)
    sigma = np.full(len(notFixIndices), 0.95, dtype=np.double)

    validRows = np.flatnonzero(hasEnoughValidNeighbors)
    if len(validRows) > 0:
        neighborVectors = (
            skeletonData["Coords"][neighbors[validRows]] - skeletonData["Coords"][notFixIndices[validRows], np.newaxis])
        neighborVectors[~neighborIsValid[validRows]] = 0.0

        covariance = np.einsum("mki,mkj->mij", neighborVectors, neighborVectors)
        eigenValues = np.linalg.eigvalsh(covariance)
        sigma[validRows] = eigenValues[:, -1] / eigenValues.sum(axis=1)

    skeletonData["Sigma"][notFixIndices] = sigma

    invSigma = 1.0 - skeletonData["Sigma"] # (n)

    neighborInvSigmaSum = np.where(neighborIsValid, invSigma[neighbors], 0.0).sum(axis=-1) # (nF)
    sigma = 1.0 - ((invSigma[notFix] + neighborInvSigmaSum) / (1.0 + validNeighborsCount))

    skeletonData["Sigma"][notFix] = np.where(sigma < 0, 0.5, sigma)


def setFixPoints(skeletonData: npt.NDArray,
                 sigmaThreshold: float):
    isBranch = skeletonData["isBranch"]
    skeletonData["isFix"][~isBranch] = skeletonData["Sigma"][~isBranch] > sigmaThreshold
