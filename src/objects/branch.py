from __future__ import annotations
import cupy as cp
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

from ..utils.geometry import linesPointsDistances


class Branch:
    def __init__(self,
                 id: int,
                 skeletonIDs: npt.NDArray[np.intp],
                 skeletonPoints: npt.NDArray[np.double],
                 tlsPoints: npt.NDArray[np.double],
                 cenOrder: int,
                 parent: Branch = None,
                 onGPU: bool = False):
        self.id = id
        self.isFlipped = False
        self.skeletonIDs = skeletonIDs
        self.skeletonPoints = skeletonPoints
        self.tlsPoints = tlsPoints
        self.cenOrder = cenOrder
        self.onGPU = onGPU
        self.children: set[Branch] = set()
        self.parent: Branch = None
        self.setParent(parent)
        self.updateSkeletonMeasurements()
        self.isRemoved = False


    def unflip(self):
        if self.isFlipped:
            self.isFlipped = False
            self.skeletonIDs = np.flipud(self.skeletonIDs)
            self.skeletonPoints = np.flipud(self.skeletonPoints)


    def orientToBifurcation(self,
                            bifurcationID: np.intp):
        if self.skeletonIDs[-1] == bifurcationID:
            self.isFlipped = not self.isFlipped
            self.skeletonIDs = np.flipud(self.skeletonIDs)
            self.skeletonPoints = np.flipud(self.skeletonPoints)
        elif not self.skeletonIDs[0] == bifurcationID:
            raise ValueError("Bifurcation can only occur at terminal branch points")


    def getInterpolation(self, resolution: np.double) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.intp]]:

        segments = self.skeletonPoints - np.roll(self.skeletonPoints, -1, axis=0)
        segmentLengths = np.linalg.norm(segments, axis=1)

        sampleCounts = np.floor(segmentLengths / resolution).astype(np.intp) + 1
        sampleCounts[-1] = 1
        indices = np.repeat(np.arange(sampleCounts.size), sampleCounts)

        interpolation = np.concat([
            np.linspace(self.skeletonPoints[i], self.skeletonPoints[i+1], sampleCounts[i], endpoint=False) for i in range(len(sampleCounts) - 1)]
            + [self.skeletonPoints[-1].reshape(1,3)])
        return interpolation, indices
    

    def getDistances(self, points: npt.NDArray[np.double]) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        xp = cp.get_array_module(points)
        segmentDistances = self.getSegmentDistances(points)
        return xp.min(segmentDistances, axis=0)


    def getSegmentDistances(self, points: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        xp = cp.get_array_module(points)
        skeletonPoints = xp.asarray(self.skeletonPoints)
        segmentDistances = linesPointsDistances(skeletonPoints[:-1], skeletonPoints[1:], points)
        return segmentDistances


    def getPartDistances(self, points: npt.NDArray[np.double]) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        xp = cp.get_array_module(points)
        segmentDistances = self.getSegmentDistances(points)
        firstSegmentDistances = segmentDistances[0]
        if len(segmentDistances) > 1:
            restBranchDistances = xp.min(segmentDistances[1:], axis=0)
        else:
            restBranchDistances = xp.full(segmentDistances.shape[1], xp.inf, dtype=xp.double)

        return firstSegmentDistances, restBranchDistances
    

    def updateSkeletonMeasurements(self):
        self.__boundingBoxMin = np.min(self.skeletonPoints, axis=0)
        self.__boundingBoxMax = np.max(self.skeletonPoints, axis=0)

        segments = self.skeletonPoints[1:] - self.skeletonPoints[:-1]
        segmentLengths = np.linalg.norm(segments)
        self.length = np.sum(segmentLengths)


    def getBoundingBoxDistance(self, boundingBoxMin, boundingBoxMax):
        delta1 = np.clip(self.__boundingBoxMin - boundingBoxMax, a_min=0, a_max=None)
        delta2 = np.clip(boundingBoxMin - self.__boundingBoxMax, a_min=0, a_max=None)
        return np.linalg.norm(delta1 + delta2)
    

    def expand(self):
        if len(self.children) != 1:
            raise ValueError("Only branches with a single child branch can be expanded")
        child = list(self.children)[0]
        if self.isFlipped or child.isFlipped:
            raise ValueError("Ensure branches are not flipped before expanding!")
        self.skeletonIDs = np.concatenate((self.skeletonIDs, child.skeletonIDs))
        self.skeletonPoints = np.concatenate((self.skeletonPoints, child.skeletonPoints))
        xp = cp.get_array_module(self.tlsPoints, child.tlsPoints)
        self.tlsPoints = xp.concatenate((self.tlsPoints, child.tlsPoints))
        self.updateSkeletonMeasurements()
        child.remove()


    def setParent(self, parent: Branch):
        if self.parent is not None:
            self.parent.children.remove(self)

        self.parent = parent

        if self.parent is not None:
            self.parent.children.add(self)


    def remove(self):
        self.isRemoved = True
        for child in self.children.copy():
            child.setParent(self.parent)
        self.setParent(None)
