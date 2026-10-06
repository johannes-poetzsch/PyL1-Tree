from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple

from src.objects.boundedObject import BoundedObject
from src.objects.branch import Branch
from src.utils.geometry import normalizeVectors


class Component(BoundedObject):
    def __init__(self, branches: List[Branch]):
        boundingBoxMin = np.min(np.stack([b.boundingBoxMin for b in branches]), axis=0)
        boundingBoxMax = np.max(np.stack([b.boundingBoxMax for b in branches]), axis=0)
        BoundedObject.__init__(self, boundingBoxMin, boundingBoxMax)

        self.branches = branches
        self.directionData = {}


    def getDirectionData(self,
                         branch: Branch,
                         minDistanceThreshold: float
                         ) -> (Tuple[Optional[npt.NDArray[np.double]], Optional[npt.NDArray[np.double]]] |
                               Tuple[int | npt.NDArray[np.double], Optional[npt.NDArray[np.double]]]):
        if branch not in self.directionData:
            self.generateDirectionData(branch, minDistanceThreshold)
        return self.directionData[branch]


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        raise NotImplementedError("only implemented by subclasses")


    @staticmethod
    def directionIsIgnored(idx_1: int | npt.NDArray[np.intp],
                           idx_2: int | npt.NDArray[np.intp],
                           branch: Branch,
                           minDistanceThreshold: float) -> bool | npt.NDArray[bool]:
        return (branch.isEndPoint(idx_1) & branch.isEndPoint(idx_2) &
                (np.linalg.norm(branch[idx_1] - branch[idx_2]) <= minDistanceThreshold))


    @staticmethod
    def getDirections(idx: int,
                      orientation: int,  # 1 or -1, determines the orientations of directions along the branch
                      branch: Branch,
                      minDistanceThreshold: float) -> List[npt.NDArray[np.double]]:
        if orientation == 1:
            endIdx = len(branch) - 1
        elif orientation == -1:
            endIdx = 0
        else:
            raise ValueError("Orientation must be 1 or -1")

        if idx == endIdx:
            return []

        directions = []
        # direction of whole branch
        if not Component.directionIsIgnored(endIdx, idx, branch, minDistanceThreshold):
            directions.append(normalizeVectors(branch[endIdx] - branch[idx]))

        nextIdx = idx + orientation
        # direction at start of branch
        if (nextIdx != endIdx and
                not Component.directionIsIgnored(nextIdx, idx, branch, minDistanceThreshold)):
            directions.append(normalizeVectors(branch[nextIdx] - branch[idx]))

        return directions
