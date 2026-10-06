from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


class BoundedObject:
    def __init__(self, boundingBoxMin: npt.NDArray[np.double], boundingBoxMax: npt.NDArray[np.double]):
        self.boundingBoxMin = boundingBoxMin
        self.boundingBoxMax = boundingBoxMax


    def getBoundingBoxDistance(self, boundingBoxMin: npt.NDArray[np.double], boundingBoxMax: npt.NDArray[np.double]) -> float:
        return BoundedObject.getBBDistance((self.boundingBoxMin, self.boundingBoxMax),
                                           (boundingBoxMin, boundingBoxMax))


    def getObjectBBDistance(self, other: BoundedObject) -> float:
        return self.getBoundingBoxDistance(other.boundingBoxMin, other.boundingBoxMax)


    def getPointsBBDistance(self, points: npt.NDArray[np.double]) -> float:
        return self.getBoundingBoxDistance(points, points)


    def getBBDistance(boundingBoxA: Tuple[npt.NDArray[np.double], npt.NDArray[np.double]],
                      boundingBoxB: Tuple[npt.NDArray[np.double], npt.NDArray[np.double]],
                      ) -> float:
        delta1 = np.clip(boundingBoxA[0] - boundingBoxB[1], a_min=0, a_max=None)
        delta2 = np.clip(boundingBoxB[0] - boundingBoxA[1], a_min=0, a_max=None)
        return np.linalg.norm(delta1 + delta2, axis=-1)
