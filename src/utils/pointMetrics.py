from multiprocessing import cpu_count, Pool

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from typing import Optional


_tlsLookUp: Optional[KDTree] = None
_searchRange: float = 0.0

def calculateADensity(point: npt.NDArray[np.double]):
    neighbors = _tlsLookUp.query_ball_point(point, _searchRange)
    if len(neighbors) <= 1: # neighborhood only contains point itself
        return 1
    else:
        distances: npt.NDArray = np.linalg.norm(_tlsLookUp.data[neighbors] - point, axis=-1)
        weights = decayingWeights(distances[distances > 0], _searchRange)
        density = 1 / (1 + weights.sum())
        return density


def calculateDensity(points: Optional[npt.NDArray[np.double]],
                     tlsLookUp: KDTree,
                     searchRange: float,
                     workers: int = 1
                     ) -> npt.NDArray[np.double]:
    if points is None:
        points = tlsLookUp.data

    global _tlsLookUp, _searchRange
    _tlsLookUp, _searchRange = tlsLookUp, searchRange

    with Pool(processes=workers) as pool:
        densities = pool.map(calculateADensity, points, chunksize=64)
    densities = np.asarray(densities, dtype=np.double)

    return densities


def decayingWeights(neighborDistances: npt.NDArray[np.double],
                    searchRange: float
                    ) -> npt.NDArray[np.double]:
    # equivalent to the formula  e^{-||x_i, p_j||^2 / (r/2)^2
    return np.exp(-np.square((2/searchRange) * neighborDistances))
