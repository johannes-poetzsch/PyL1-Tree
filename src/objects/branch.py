import cupy as cp
import numpy as np
import numpy.typing as npt
from typing import Tuple

from ..utils.geometry import linesPointsDistances


class Branch:
    def __init__(self,
                 ID: np.intp,
                 skeletonIDs: npt.NDArray[np.intp],
                 skeletonPoints: npt.NDArray[np.double],
                 tlsPoints: npt.NDArray[np.double],
                 cenOrder: int,
                 onGPU: bool = False):
        self.ID = ID
        self.isFlipped = False
        self.skeletonIDs = skeletonIDs
        self.skeletonPoints = skeletonPoints
        self.tlsPoints = tlsPoints
        self.cenOrder = cenOrder
        self.onGPU = onGPU


    def toGPU(self):
        if not self.onGPU:
            self.onGPU = True
            self.tlsPoints = cp.asarray(self.tlsPoints)


    def toCPU(self):
        if self.onGPU:
            self.onGPU = False
            self.tlsPoints = cp.asnumpy(self.tlsPoints)


    def unflip(self):
        if self.isFlipped():
            self.isFlipped = False
            self.skeletonIDs = np.flipud(self.skeletonIDs)
            self.skeletonPoints = np.flipud(self.skeletonPoints, axis=0)


    def orientToBifurcation(self,
                            bifurcationID: np.intp):
        if self.skeletonIDs[-1] == bifurcationID:
            self.isFlipped = not self.isFlipped
            self.skeletonIDs = np.flipud(self.skeletonIDs)
            self.skeletonPoints = np.flipud(self.skeletonPoints)
        elif not self.skeletonIDs[0] == bifurcationID:
            raise ValueError("Bifurcation can only occur at terminal branch points")


    def getInterpolation(self, resolution: np.double) -> npt.NDArray[np.double]:

        segments = self.skeletonPoints - np.roll(self.skeletonPoints, 1, axis=0)
        segmentLengths = np.linalg.norm(segments, axis=1)

        sampleCounts = np.floor(segmentLengths / resolution).astype(np.intp) + 1
        sampleCounts[0] = 1
        interpolation = np.concat([
            np.linspace(self.skeletonPoints[i], self.skeletonPoints[i+1], sampleCounts[i+1], endpoint=False) for i in range(len(sampleCounts) - 1)] 
            + [self.skeletonPoints[-1].reshape(1,3)])
        return interpolation


    def getPartDistances(self, points: npt.NDArray[np.double]) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        xp = cp.get_array_module(points)
        skeletonPoints = xp.asarray(self.skeletonPoints)
        segmentDistances = linesPointsDistances(skeletonPoints[:-1],
                                                xp.roll(skeletonPoints, -1, axis=0)[:-1],
                                                points)
        firstSegmentDistances = segmentDistances[0]
        restBranchDistances = xp.min(segmentDistances[1:], axis=0)

        return firstSegmentDistances, restBranchDistances
