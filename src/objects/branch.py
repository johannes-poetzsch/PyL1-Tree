from __future__ import annotations
import cupy as cp
from itertools import count, repeat
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Iterable, List, Optional, Tuple

from src.objects.boundedObject import BoundedObject
from src.utils.geometry import getStripAngles, getValueOccurrences, linesPointsDistances, stripPointsDistances


class Branch(BoundedObject):
    idGenerator = count()

    def __init__(self,
                 skeletonPoints: npt.NDArray[np.double],
                 tlsPoints: Optional[npt.NDArray[np.double]] = None,
                 parent: Branch = None):
        BoundedObject.__init__(self, None, None)

        self.id = next(Branch.idGenerator)
        self.isFlipped = False
        self.skeletonPoints = skeletonPoints
        self.trackEndPoints = False
        self.endPointIndices: Optional[List[int]] = None
        self.temporaryIndices: Optional[List[int]] = None
        self.tlsPoints = np.empty([0,3], dtype=np.double) if tlsPoints is None else tlsPoints
        self.children: set[Branch] = set()

        self.parent: Optional[Branch] = None
        self.setParent(parent)

        self.length = None
        self.updateSkeletonMeasurements()

        self.isRemoved = False


    def __len__(self):
        return len(self.skeletonPoints)


    def __getitem__(self, item):
        return self.skeletonPoints[item]


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


    def getCentrifugalOrder(self) -> int:
        if self.isStem():
            return 1
        else:
            return self.parent.getCentrifugalOrder() + 1


    def addEndPoints(self, endPointIndices: Iterable[int]):
        self.trackEndPoints = True
        if self.endPointIndices is None:
            self.endPointIndices = [0, len(self.skeletonPoints) - 1]
        self.endPointIndices += list(endPointIndices)


    def isEndPoint(self, idx: int | npt.NDArray[int]) -> bool | npt.NDArray[bool]:
        idx = idx % len(self.skeletonPoints)
        if self.endPointIndices is None:
            return (idx == 0) | (idx == len(self.skeletonPoints) - 1)
        else:
            return np.isin(idx, self.endPointIndices)


    def flip(self):
        self.isFlipped = not self.isFlipped
        self.skeletonPoints = np.flipud(self.skeletonPoints)


    def unflip(self):
        if self.isFlipped:
            self.isFlipped = False
            self.skeletonPoints = np.flipud(self.skeletonPoints)


    def getInterpolation(self, resolution: float) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.intp]]:
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


    def getAngles(self) -> Optional[npt.NDArray[np.double]]:
        return getStripAngles(self.skeletonPoints)
    

    def getDistances(self, 
                     points: npt.NDArray[np.double],
                     trimStart = False,
                     trimEnd = False,
                     returnTrimming = False
                     ) -> (npt.NDArray[np.double] |
                           Tuple[npt.NDArray[np.double], npt.NDArray[np.bool]]):
        return stripPointsDistances(self.skeletonPoints, points, trimStart, trimEnd, returnTrimming)


    def interpolate(self, resolution: float, temporary: bool = False):
        interpolation, interpolationIndices = self.getInterpolation(resolution)
        _, originalIndices = np.unique(interpolationIndices, return_index=True)

        self.skeletonPoints = interpolation

        if self.trackEndPoints and self.endPointIndices is not None:
            self.endPointIndices = originalIndices[self.endPointIndices]

        if temporary:
            self.temporaryIndices = [idx for idx in range(len(self.skeletonPoints)) if idx not in originalIndices]


    def removeTemporaryPoints(self):
        if self.temporaryIndices is None:
            return
        originalIndices = [idx for idx in range(len(self.skeletonPoints)) if idx not in self.temporaryIndices]
        self.skeletonPoints = self.skeletonPoints[originalIndices]

        if self.trackEndPoints and self.endPointIndices is not None:
            self.endPointIndices = [newIdx for newIdx, idx in enumerate(originalIndices) if self.isEndPoint(idx)]

        self.temporaryIndices = None


    def updateSkeletonMeasurements(self):
        self.boundingBoxMin = np.min(self.skeletonPoints, axis=0)
        self.boundingBoxMax = np.max(self.skeletonPoints, axis=0)

        segments = self.skeletonPoints[1:] - self.skeletonPoints[:-1]
        self.length = np.linalg.norm(segments, axis=-1).sum()


    def expand(self) -> Branch:
        if len(self.children) != 1:
            raise ValueError("Only branches with a single child branch can be expanded")
        removedChild = next(iter(self.children))

        if self.isFlipped or removedChild.isFlipped:
            raise ValueError("Ensure branches are not flipped before expanding!")

        originalPointCount = len(self)
        self.tlsPoints = np.append(self.tlsPoints, removedChild.remove(adjustConnectedBranches=False), axis=0)
        self.skeletonPoints = np.append(self.skeletonPoints, removedChild.skeletonPoints[1:], axis=0)
        self.updateSkeletonMeasurements()

        if self.trackEndPoints:
            self.addEndPoints([originalPointCount-1])
            if removedChild.endPointIndices is not None:
                self.addEndPoints([len(self.skeletonPoints)-1 + idx for idx in removedChild.endPointIndices
                                   if idx not in [0, len(removedChild.skeletonPoints) - 1]])
        if removedChild.temporaryIndices is not None:
            if self.temporaryIndices is None:
                self.temporaryIndices = []
            self.temporaryIndices.extend([idx + originalPointCount - 1 for idx in removedChild.temporaryIndices])

        return removedChild


    def remove(self, adjustConnectedBranches: bool = True) -> npt.NDArray[np.double]:
        self.isRemoved = True
        parent = self.parent
        self.setParent(None)

        for child in list(self.children):
            if adjustConnectedBranches:
                assert(not self.isFlipped)
                assert(not child.isFlipped)
                child.skeletonPoints[0] = self.skeletonPoints[0]
                child.updateSkeletonMeasurements()
            child.setParent(parent)

        if adjustConnectedBranches and not parent is None and len(parent.children) == 1:
            parent.expand()

        return self.tlsPoints


    def reorientSubtree(self, newParent: Branch):
        for child in list(self.children):
            if child is not newParent:
                child.setParent(newParent)

        self.skeletonPoints = np.flipud(self.skeletonPoints)
        if self.trackEndPoints and self.endPointIndices is not None:
            self.endPointIndices = [len(self.skeletonPoints)-1 - idx for idx in self.endPointIndices]
        if self.temporaryIndices is not None:
            self.temporaryIndices = [len(self.temporaryIndices) - 1 - idx for idx in self.temporaryIndices]

        if self.parent is not None:
            self.parent.reorientSubtree(self)
        self.setParent(newParent)


    # returns branch that ends at splitIdx.
    def split(self, splitIdx: int | np.intp) -> Tuple[Optional[Branch], Optional[Branch]]:
        splitIdx = splitIdx % len(self.skeletonPoints)

        if splitIdx == 0:
            return self.parent, None
        if splitIdx == len(self.skeletonPoints) - 1:
            return self, None

        splitEnd = Branch(self.skeletonPoints[splitIdx:], parent=self)

        for child in [c for c in self.children if c is not splitEnd]:
            child.setParent(splitEnd)

        self.skeletonPoints = self.skeletonPoints[:splitIdx + 1]
        if len(self.tlsPoints) > 0:
            Branch.reassignPoints([self, splitEnd], affectedBranches=[self])

        if self.trackEndPoints and self.endPointIndices is not None:
            endPointIndices = self.endPointIndices
            self.endPointIndices = None
            for idx in [i for i in endPointIndices if i not in [0, splitIdx, len(self.skeletonPoints) - 1]]:
                if idx < splitIdx:
                    self.addEndPoints([idx])
                else:
                    splitEnd.addEndPoints([idx - splitIdx])

        if self.temporaryIndices is not None:
            splitEnd.temporaryIndices = [idx - splitIdx for idx in self.temporaryIndices if idx > splitIdx]
            self.temporaryIndices = [idx for idx in self.temporaryIndices if idx < splitIdx]

        self.updateSkeletonMeasurements()

        return self, splitEnd


    @staticmethod
    def assignAllPoints(points: npt.NDArray[np.double],
                        branches: List[Branch],
                        useTrimming=False) -> npt.NDArray[np.double]:
        closeBranchIndices = Branch.findCloseBranches(points, branches)
        unassignedPoints = []
        for branchIdx, indices in getValueOccurrences(closeBranchIndices):
            unassignedPoints.append(
                Branch.assignLocalPoints(points[indices], branches, [branches[branchIdx]],
                                         useTrimming=useTrimming))
        return np.concatenate(unassignedPoints, axis=0)


    @staticmethod
    def assignLocalPoints(points: npt.NDArray[np.double],
                          branches: List[Branch],
                          closeBranches: Optional[List[Branch]] = None,
                          originalDistances: Optional[npt.NDArray[np.double]] = None,
                          useTrimming=False) -> npt.NDArray[np.double]:
        closeBranches = branches if closeBranches is None else closeBranches
        closestBranchIndices, _ = Branch.findLocalClosestBranches(points, branches, closeBranches,
                                                                  originalDistances, useTrimming)

        if cp.get_array_module(points) is cp:
            points = cp.asnumpy(points)

        unassignedPoints = np.empty((0, 3), dtype=points.dtype)
        for branchIdx, indices in getValueOccurrences(closestBranchIndices):
            if branchIdx < 0:
                unassignedPoints = points[indices]
            else:
                branch = branches[branchIdx]
                branch.tlsPoints = np.append(branch.tlsPoints, points[indices], axis=0)

        return unassignedPoints


    @staticmethod
    def findCloseBranches(points: npt.NDArray[np.double],
                          branches: List[Branch]) -> npt.NDArray[int]:
        skeletonPoints = np.concatenate([b.skeletonPoints for b in branches], axis=0)
        branchIds = np.arange(len(branches), dtype=int).repeat([len(b) for b in branches])

        lookUp = KDTree(skeletonPoints)

        _, closestIndices = lookUp.query(points, k=1)
        closeBranchIndices = branchIds[closestIndices]

        return closeBranchIndices


    @staticmethod
    def findClosestBranches(points: npt.NDArray[np.double],
                            branches: List[Branch],
                            branchIndices: Optional[Iterable[int]] = None,
                            initialClosest: Optional[npt.NDArray[np.intp]] = None,
                            initialDistances: Optional[npt.NDArray[np.double]] = None,
                            useTrimming = False
                            ) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.double]]:
        if len(points) == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.double)
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
                newDistances, trimmed = branch.getDistances(points, trimStart=branch.isStem(), trimEnd=branch.isLeaf(),
                                                            returnTrimming=True)
            else:
                newDistances = branch.getDistances(points)
            improved = newDistances < distances

            if xp.count_nonzero(improved) > 0:
                distances[improved] = newDistances[improved]
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
    def findGlobalClosestBranches(points: npt.NDArray[np.double],
                                  branches: List[Branch]
                                  ) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.double]]:
        closeBranchIndices = Branch.findCloseBranches(points, branches)

        minDistances = np.empty_like(closeBranchIndices, dtype=np.double)
        closestIndices = np.empty_like(closeBranchIndices, dtype=np.intp)

        for branchIdx, indices in getValueOccurrences(closeBranchIndices):
            closestIndices[indices], minDistances[indices] = Branch.findLocalClosestBranches(
                points[indices], branches, [branches[branchIdx]], useTrimming=False)

        return closestIndices, minDistances


    @staticmethod
    def reassignPoints(branches: List[Branch],
                       unassignedTlsPoints: Optional[npt.NDArray[np.double]] = None,
                       affectedBranches: Optional[List[Branch]] = None,
                       originalDistances: Optional[List[npt.NDArray[np.double]]] = None,
                       useTrimming: bool = True
                       ) -> npt.NDArray[np.double]:
        affectedBranches = branches if affectedBranches is None else affectedBranches
        originalDistances = repeat(None) if originalDistances is None else originalDistances
        pointsToReassign = [b.tlsPoints for b in affectedBranches]
        for b in affectedBranches:
            b.tlsPoints = np.empty((0,3), dtype=np.double)

        prunedTlsPoints = []
        for branch, tlsPoints, od in zip(affectedBranches, pointsToReassign, originalDistances):
            if len(tlsPoints) == 0:
                continue
            prunedTlsPoints.append(Branch.assignLocalPoints(tlsPoints, branches, closeBranches=[branch],
                                                            originalDistances=od,
                                                            useTrimming=useTrimming))

        if unassignedTlsPoints is not None:
            prunedTlsPoints.append(Branch.assignAllPoints(unassignedTlsPoints, branches, useTrimming=True))

        if len(prunedTlsPoints) > 0:
            return np.concatenate(prunedTlsPoints, axis=0)
        else:
            return np.empty((0,3), dtype=np.double)
