from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from src.utils.pointMetrics import decayingWeights


_baseRadius: Optional[float] = None
_searchRange: Optional[float] = None
_skeletonData: Optional[npt.NDArray] = None
_tlsDensities: Optional[npt.NDArray[np.double]] = None
_skeletonLookUp: Optional[KDTree] = None
_tlsLookUp: Optional[KDTree] = None


def setIterationParameters(baseRadius: float,
                           searchRange: float,
                           skeletonData: npt.NDArray,
                           tlsDensities: npt.NDArray[np.double],
                           tlsLookUp: KDTree):
    global _baseRadius, _searchRange, _skeletonData, _tlsDensities, _skeletonLookUp, _tlsLookUp

    _baseRadius = baseRadius
    _searchRange = searchRange
    _skeletonData = skeletonData
    _tlsDensities = tlsDensities
    _skeletonLookUp = KDTree(skeletonData["Coords"])
    _tlsLookUp = tlsLookUp


def runIteration(idx: int):
    neighborRadius = np.maximum(2 * _searchRange, 2 * _baseRadius * _skeletonData["relPathLocation"][idx])
    neighbors = _skeletonLookUp.query_ball_point(_skeletonData["Coords"][idx], neighborRadius)

    if len(neighbors) <= 1:
        return idx, False, 0.5, None, None

    neighbors = [n for n in neighbors if n != idx]
    neighborVectors = _skeletonData["Coords"][idx] - _skeletonData["Coords"][neighbors]
    neighborDistances = np.linalg.norm(neighborVectors, axis=-1)
    _sigma = calculateDirectionalDegree(neighborVectors, neighborDistances, _searchRange)
    _repulsion = calculateRepulsion(neighborVectors, neighborDistances, _searchRange)

    tlsNeighbors = _tlsLookUp.query_ball_point(_skeletonData["Coords"][idx], neighborRadius)
    if len(tlsNeighbors) == 0:
        return idx, False, _sigma, None, None

    _tlsAverage = calculatePointAverage(_skeletonData["Coords"][idx], _tlsLookUp.data[tlsNeighbors],
                                        _tlsDensities[tlsNeighbors], _searchRange)
    return idx, True, _sigma, _repulsion, _tlsAverage


def calculateDirectionalDegree(neighborVectors: npt.NDArray[np.double],
                               neighborDistances: Optional[npt.NDArray[np.double]] = None,
                               searchRange: Optional[float] = None,
                               useDecayingWeights: bool = True) -> np.double:
    if len(neighborVectors) <= 3:
        return np.double(0.5)

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
