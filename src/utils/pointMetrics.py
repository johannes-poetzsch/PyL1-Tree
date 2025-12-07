import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from typing import Optional


def calculateDensity(points: Optional[npt.NDArray[np.double]],
                     tlsLookUp: KDTree,
                     searchRange: float
                     ) -> npt.NDArray[np.double]:
    if points is None:
        points = tlsLookUp.data
        neighbors = tlsLookUp.query_ball_tree(tlsLookUp, searchRange)
    else:
        neighbors = tlsLookUp.query_ball_point(points, searchRange)

    densities = np.empty(len(points), dtype=np.double)
    for idx, neighborList in enumerate(neighbors):
        if len(neighborList) <= 1: # neighborhood only contains point itself
            densities[idx] = 1
        else:
            distances: npt.NDArray = np.linalg.norm(tlsLookUp.data[neighborList] - points[idx], axis=-1)
            weights = decayingWeights(distances[distances > 0], searchRange)
            densities[idx] = 1 / (1 + weights.sum())

    return densities


def decayingWeights(neighborDistances: npt.NDArray[np.double],
                    searchRange: float
                    ) -> npt.NDArray[np.double]:
    # equivalent to the formula  e^{-||x_i, p_j||^2 / (r/2)^2
    return np.exp(-np.square((2/searchRange) * neighborDistances))
