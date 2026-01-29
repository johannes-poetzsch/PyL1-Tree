from __future__ import annotations
import cupy as cp
import numpy as np
import numpy.typing as npt
from typing import Iterable, List, Optional, Tuple

from ..utils.branchUtils import findCloseBranches
from ..utils.geometry import getStripAngles, linesPointsDistances, stripPointsDistances


class Branch:
    def __init__(self,
                 id: int,
                 skeletonIDs: npt.NDArray[np.intp],
                 skeletonPoints: npt.NDArray[np.double],
                 tlsPoints: npt.NDArray[np.double] = np.empty([0,3], dtype=np.double),
                 cenOrder: int = None,
                 parent: Branch = None,
                 onGPU: bool = False):
        self.id = id
        self.isFlipped = False
        self.skeletonIDs = skeletonIDs # TODO: can probably be removed
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
        

    # returns branch that ends at splitIdx.
    def split(self, splitIdx: int) -> Branch:
        if splitIdx <= 0:
            return self.parent
        if splitIdx >= len(self.skeletonPoints) - 1:
            return self

        splitEnd = Branch(self.id, self.skeletonIDs[splitIdx:], self.skeletonPoints[splitIdx:])

        for child in list(self.children):
            child.setParent(splitEnd)

        splitEnd.setParent(self)
        self.skeletonIDs = self.skeletonIDs[:splitIdx+1]
        self.skeletonPoints = self.skeletonPoints[:splitIdx+1]
        self.updateSkeletonMeasurements()

        return self


    def getInterpolation(self, resolution: np.double) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.intp]]:

        segments = self.skeletonPoints[1:] - self.skeletonPoints[:-1]
        segmentLengths = np.linalg.norm(segments, axis=1)

        sampleCounts = np.floor(segmentLengths / resolution).astype(np.intp) + 1

        indices = np.repeat(np.arange(len(sampleCounts)), sampleCounts)
        indices = np.append(indices, len(self.skeletonPoints) - 1)

        interpolation = np.concat([
            np.linspace(self.skeletonPoints[i], self.skeletonPoints[i+1], sampleCounts[i], endpoint=False) for i in range(len(sampleCounts))]
            + [self.skeletonPoints[-1].reshape(1,3)], 
            axis=0)
        return interpolation, indices
    

    def getDistances(self, 
                     points: npt.NDArray[np.double],
                     trimStart = False,
                     trimEnd = False,
                     returnTrimming = False
                     ) -> (npt.NDArray[np.double] |
                           Tuple[npt.NDArray[np.double], npt.NDArray[np.bool]]):
        return stripPointsDistances(self.skeletonPoints, points, trimStart, trimEnd, returnTrimming)


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
    

    def getAngles(self) -> Optional[npt.NDArray[np.double]]:
        return getStripAngles(self.skeletonPoints)
    

    def updateSkeletonMeasurements(self):
        self.boundingBoxMin = np.min(self.skeletonPoints, axis=0)
        self.boundingBoxMax = np.max(self.skeletonPoints, axis=0)

        segments = self.skeletonPoints[1:] - self.skeletonPoints[:-1]
        self.length = np.linalg.norm(segments, axis=-1).sum()


    def getBoundingBoxDistance(self, boundingBoxMin: np.double, boundingBoxMax: np.double):
        delta1 = np.clip(self.boundingBoxMin - boundingBoxMax, a_min=0, a_max=None)
        delta2 = np.clip(boundingBoxMin - self.boundingBoxMax, a_min=0, a_max=None)
        return np.linalg.norm(delta1 + delta2, axis=-1)
    

    def branchBoundingBoxDistance(self, other: Branch):
        return self.getBoundingBoxDistance(other.boundingBoxMin, other.boundingBoxMax)
    

    def expand(self):
        if len(self.children) != 1:
            raise ValueError("Only branches with a single child branch can be expanded")
        child = self.children.pop()
        if self.isFlipped or child.isFlipped:
            raise ValueError("Ensure branches are not flipped before expanding!")
        self.skeletonIDs = np.append(self.skeletonIDs, child.skeletonIDs[1:], axis=0)
        self.skeletonPoints = np.append(self.skeletonPoints, child.skeletonPoints[1:], axis=0)
        xp = cp.get_array_module(self.tlsPoints, child.tlsPoints)
        self.tlsPoints = xp.append(self.tlsPoints, child.tlsPoints, axis=0)
        self.updateSkeletonMeasurements()


    def isLeaf(self):
        return len(self.children) == 0
    

    def isStem(self):
        return self.parent is None


    def setParent(self, parent: Optional[Branch]):
        if self.parent is not None:
            self.parent.children.remove(self)

        if parent is not None:
            parent.children.add(self)

        self.parent = parent


    def remove(self, adjustConnectedBranches = True) -> npt.NDArray[np.double]:
        self.isRemoved = True
        for child in self.children.copy():
            if adjustConnectedBranches:
                assert(not self.isFlipped)
                assert(not child.isFlipped)
                child.skeletonIDs[0] = self.skeletonIDs[0]
                child.skeletonPoints[0] = self.skeletonPoints[0]
                child.updateSkeletonMeasurements()
            child.setParent(self.parent)

        if adjustConnectedBranches and len(self.parent.children) == 1:
            self.parent.expand()

        self.setParent(None)

        return self.tlsPoints


    @staticmethod
    def findClosestBranches(points: npt.NDArray[np.double],
                            branches: List[Branch],
                            branchIndices: Optional[Iterable[int]] = None,
                            initialClosest: Optional[npt.NDArray[np.intp]] = None,
                            initialDistances: Optional[npt.NDArray[np.double]] = None,
                            useTrimming = False
                            ) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.double]]:
        xp = cp.get_array_module(points)
        pointsMin = points.min(axis=0)
        pointsMax = points.max(axis=0)

        branchIndices = range(len(branches)) if branchIndices is None else branchIndices

        if initialDistances is None or initialClosest is None:
            distances = xp.full(len(points), xp.inf, dtype=xp.double)
            closest = xp.full(len(points), -1, dtype=int)
        else:
            distances = xp.asarray(initialDistances)
            closest = initialClosest

        maxDistance = distances.max()

        for idx in branchIndices:
            branch = branches[idx]
            if branch.getBoundingBoxDistance(pointsMin, pointsMax) >= maxDistance:
                continue

            if useTrimming:
                dist, trimmed = branch.getDistances(points, trimStart=branch.isStem(), trimEnd=branch.isLeaf(),
                                                    returnTrimming=True)
            else:
                dist = branch.getDistances(points)
            improved = dist < distances

            if xp.count_nonzero(improved) > 0:
                distances[improved] = dist[improved]
                maxDistance = distances.max()

                if branch.isRemoved:
                    closest[improved] = -1
                else:
                    closest[improved] = idx
                    if useTrimming:
                        closest[improved & trimmed] = -1

        if xp is cp:
            closest, distances = cp.asnumpy(closest), cp.asnumpy(distances)
        
        return closest, distances


    @staticmethod
    def findLocalClosestBranches(points: npt.NDArray[np.double],
                                 branches: List[Branch],
                                 closeBranches: List[Branch],
                                 originalDistances: Optional[npt.NDArray[np.double]] = None,
                                 useTrimming = False
                                 ) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.double]]:
        xp = cp.get_array_module(points)

        closeBranches = set(closeBranches)
        closeBranchIndices, otherBranchIndices = [], []
        for idx in range(len(branches)):
            (closeBranchIndices if branches[idx] in closeBranches else otherBranchIndices).append(idx)

        closeIndices, distances = Branch.findClosestBranches(points, branches,
                                                             branchIndices=closeBranchIndices,
                                                             useTrimming=useTrimming)

        if originalDistances is not None:
            worsened = xp.asarray(distances > originalDistances)
            closestIndices = closeIndices
            closestIndices[worsened], distances[worsened] = Branch.findClosestBranches(
                points[worsened], branches,
                branchIndices=otherBranchIndices,
                initialClosest=closeIndices[worsened],
                initialDistances=distances[worsened],
                useTrimming=useTrimming)
        else:
            closestIndices, distances = Branch.findClosestBranches(points, branches,
                                                                   branchIndices=otherBranchIndices,
                                                                   initialClosest=closeIndices,
                                                                   initialDistances=distances,
                                                                   useTrimming=useTrimming)

        return closestIndices, distances


    @staticmethod
    def assignLocalPoints(points: npt.NDArray[np.double],
                          branches: List[Branch],
                          closeBranches: List[Branch],
                          originalDistances: Optional[npt.NDArray[np.double]] = None,
                          useTrimming = False) -> npt.NDArray[np.double]:
        closestBranchIndices, _ = Branch.findLocalClosestBranches(points, branches, closeBranches,
                                                                  originalDistances, useTrimming)

        if cp.get_array_module(points) is cp:
            points = cp.asnumpy(points)

        branchOrder = np.argsort(closestBranchIndices)
        points = points[branchOrder]
        closestBranchIndices = closestBranchIndices[branchOrder]

        branchIndices, indices, counts = np.unique(closestBranchIndices, return_index=True, return_counts=True)

        unassignedPoints = np.empty((0,3), dtype=points.dtype)

        for branchIdx, idx, count in zip(branchIndices, indices, counts):
            if branchIdx < 0:
                unassignedPoints = points[idx:idx+count]
                continue
            branch = branches[branchIdx]
            branch.tlsPoints = np.append(branch.tlsPoints, points[idx:idx+count], axis=0)

        return unassignedPoints


    @staticmethod
    def assignAllPoints(points: npt.NDArray[np.double],
                        branches: List[Branch],
                        useTrimming = False) -> npt.NDArray[np.double]:
        closeBranchIndices = findCloseBranches(points, branches)

        branchOrder = np.argsort(closeBranchIndices)
        closeBranchIndices = closeBranchIndices[branchOrder]
        points = points[branchOrder]

        branchIndices, indices, counts = np.unique(closeBranchIndices, return_index=True, return_counts=True)

        unassignedPoints = []
        for branchIdx, idx, count in zip(branchIndices, indices, counts):
            unassignedPoints.append(
                Branch.assignLocalPoints(points[idx:idx+count], branches, [branches[branchIdx]],
                                         useTrimming=useTrimming))
        return np.concatenate(unassignedPoints, axis=0)
