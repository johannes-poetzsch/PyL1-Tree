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

    neighborRadii = np.maximum(2 * searchRange, skeletonData["relPathLocation"][notFix] * 2 * baseRadius)
    neighborLists = skeletonLookUp.query_ball_point(coords[notFix], neighborRadii)
    hasNeighbors = np.array([len(n) > 1 for n in neighborLists], dtype=bool)

    changing = notFix.copy()
    changing[notFix] = hasNeighbors
    changingToNotFix = np.nonzero(hasNeighbors)[0]
    changingToOriginal = np.nonzero(changing)[0]
    changingCount = len(changingToOriginal)

    sigma = np.empty(changingCount)
    repulsion = np.empty([changingCount, 3])
    for idx, data in enumerate(skeletonData[changing]):
        originalIdx = changingToOriginal[idx]
        notFixIdx = changingToNotFix[idx]

        neighbors = [n for n in neighborLists[notFixIdx] if n != originalIdx]

        neighborVectors = data["Coords"] - coords[neighbors]
        neighborDistances = np.linalg.norm(neighborVectors, axis=-1)

        sigma[idx] = calculateDirectionalDegree(neighborVectors, neighborDistances, searchRange)
        repulsion[idx] = calculateRepulsion(neighborVectors, neighborDistances, searchRange)

    tlsNeighborLists = tlsLookUp.query_ball_point(coords[changing], neighborRadii[hasNeighbors])
    tlsAverage = np.empty([changingCount, 3])
    for idx, data in enumerate(skeletonData[changing]):
        tlsNeighbors = tlsNeighborLists[idx]
        neighborDensities = tlsDensities[tlsNeighbors]
        tlsNeighborPoints = tlsLookUp.data[tlsNeighbors]

        tlsAverage[idx] = calculatePointAverage(data["Coords"], tlsNeighborPoints, neighborDensities, searchRange)

    minSigma = sigma.min()
    sigmaRange = sigma.max() - minSigma
    muRange = np.abs(maxMu - minMu)

    mu = (muRange / sigmaRange) * (sigma - minSigma) + minMu
    coords[changing] = tlsAverage + mu[:,np.newaxis] * repulsion


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
    notFixIndices = np.nonzero(notFix)[0] # (nF)

    _, neighbors = skeletonLookUp.query(skeletonData[notFix]["Coords"], kNeighbors + 1) # (nF,k+1)
    neighbors = neighbors[:,1:] # (nF,k)

    neighborExists = neighbors < len(skeletonData) # (nF,k)
    neighbors[~neighborExists] = 0

    neighborIsValid = np.logical_and(neighborExists, notFix[neighbors]) # (nF,k)
    validNeighborsCount = np.count_nonzero(neighborIsValid, axis=1) # (nF)

    hasEnoughNeighbors = np.count_nonzero(neighborExists, axis=-1) > 3 # (nF)
    hasEnoughValidNeighbors = validNeighborsCount > 3 # (nF)
    skeletonData["Sigma"][notFixIndices[~hasEnoughNeighbors]] = 0.5
    skeletonData["Sigma"][notFixIndices[hasEnoughNeighbors & ~hasEnoughValidNeighbors]] = 0.95

    neighborVectors = (skeletonData["Coords"][neighbors[hasEnoughValidNeighbors]] -
                       skeletonData["Coords"][notFixIndices[hasEnoughValidNeighbors]][:,np.newaxis,:]) # (V,k,3)
    neighborVectors = np.where(neighborIsValid[hasEnoughValidNeighbors,:,np.newaxis],
                               neighborVectors, np.zeros_like(neighborVectors))

    skeletonData["Sigma"][notFixIndices[hasEnoughValidNeighbors]] = calculateDirectionalDegree(neighborVectors,
                                                                                               useDecayingWeights=False)
    invSigma = 1.0 - skeletonData["Sigma"] # (n)

    neighborInvSigmaSum = np.where(neighborIsValid, invSigma[neighbors], 0.0).sum(axis=-1) # (nF)
    skeletonData["Sigma"][notFix] = 1.0 - ((invSigma[notFix] + neighborInvSigmaSum) / (1.0 + validNeighborsCount))

    skeletonData["Sigma"][skeletonData["Sigma"] < 0.0] = 0.5


def setFixPoints(skeletonData: npt.NDArray,
                 sigmaThreshold: float):
    isBranch = skeletonData["isBranch"]
    skeletonData["isFix"][~isBranch] = skeletonData["Sigma"][~isBranch] > sigmaThreshold
