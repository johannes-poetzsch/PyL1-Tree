from __future__ import annotations
import numpy as np
import numpy.typing as npt


class BoundedObject:
    def __init__(self, boundingBoxMin: npt.NDArray[np.double], boundingBoxMax: npt.NDArray[np.double]):
        self.boundingBoxMin = boundingBoxMin
        self.boundingBoxMax = boundingBoxMax


    def getBoundingBoxDistance(self, boundinBoxMin: npt.NDArray[np.double], boundingBoxMax: npt.NDArray[np.double]) -> float:
        delta1 = np.clip(self.boundingBoxMin - boundingBoxMax, a_min=0, a_max=None)
        delta2 = np.clip(boundinBoxMin - self.boundingBoxMax, a_min=0, a_max=None)
        return np.linalg.norm(delta1 + delta2, axis=-1)


    def getObjectBBDistance(self, other: BoundedObject) -> float:
        return self.getBoundingBoxDistance(other.boundingBoxMin, other.boundingBoxMax)


    def getPointBBDistance(self, point: npt.NDArray[np.double]) -> float:
        return self.getBoundingBoxDistance(point, point)
