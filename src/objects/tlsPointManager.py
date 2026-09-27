import numpy as np
import numpy.typing as npt
from cupyx.scipy.spatial import distance
from scipy.spatial import KDTree
from typing import List, Optional, Tuple

from src.objects.branch import Branch
from utils.geometry import queryPointsNearSegment


class TlsPointManager:
    def __init__(self,
                 branches: List[Branch],
                 unassignedPoints: npt.NDArray[np.double]):
        self.branches = branches
        self.assignedPoints = np.concatenate([b.tlsPoints for b in branches], axis=0)
        self.assignedPointsDistances = np.concatenate([b.getDistances(b.tlsPoints) for b in branches], axis=0)
        self.assignedPointsBranchIndices = np.repeat(np.arange(len(branches)), [len(b.tlsPoints) for b in branches])
        self.branchIds = {b: idx for idx, b in enumerate(branches)}
        self.assignedPointsLookUp = KDTree(self.assignedPoints)
        self.assignedPointsMaxDistance = self.assignedPointsDistances.max()

        self.unassignedPoints = unassignedPoints
        _, self.unassignedPointsDistances = Branch.findGlobalClosestBranches(self.unassignedPoints, list(branches))
        self.unassignedPointsLookUp = KDTree(self.unassignedPoints)
        self.unassignedPointsMaxDistance = self.unassignedPointsDistances.max()

        self.branchPointIds = {}
        self.searchRanges = {}
        offset = 0
        for branch in branches:
            self.branchPointIds[branch] = offset + np.arange(len(branch.tlsPoints), dtype=np.intp)
            offset += len(branch.tlsPoints)
            branch.tlsPoints = np.empty((0,3), dtype=np.double)


    def releasePoints(self) -> npt.NDArray[np.double]:
        unassignedPoints = []
        branches = [b for b in self.branches if not b.isRemoved]
        for branch in branches:
            branch.endPointIndices = None
            branch.trackEndPoints = False
            unassignedPoints.append(Branch.assignLocalPoints(self.assignedPoints[self.branchPointIds[branch]],
                                                             [branch], useTrimming=True))

        unassignedPoints.append(Branch.assignAllPoints(self.unassignedPoints, branches, useTrimming=True))

        unassignedPoints = np.concatenate(unassignedPoints, axis=0)
        return unassignedPoints


    def addBranch(self, branch: Branch):
        self.branchIds[branch] = len(self.branches)
        self.branches.append(branch)
        self.branchPointIds[branch] = np.empty((0,3), dtype=np.intp)


    def createBranch(self,
                     skeletonPoints: npt.NDArray[np.double],
                     parent: Branch,
                     assignedTlsIds: npt.NDArray[np.intp],
                     newAssignedDistances: npt.NDArray[np.double],
                     unassignedTlsIds: npt.NDArray[np.intp],
                     newUnassignedDistances: npt.NDArray[np.double],
                     resolution: float
                     ) -> Branch:
        branch = Branch(skeletonPoints, parent=parent)
        Branch.trackEndPoints = True
        self.addBranch(branch)
        branchId = self.branchIds[branch]

        self.assignedPointsDistances[assignedTlsIds] = newAssignedDistances
        changedBranchesIndices = np.unique(self.assignedPointsBranchIndices[assignedTlsIds])
        self.assignedPointsBranchIndices[assignedTlsIds] = branchId
        self.branchPointIds[branch] = assignedTlsIds

        self.unassignedPointsDistances[unassignedTlsIds] = newUnassignedDistances

        for idx in changedBranchesIndices:
            self.searchRanges[self.branches[idx]] = None

        branch.interpolate(resolution, temporary=True)
        return branch


    def getDistances(self, branch: Branch) -> npt.NDArray[np.double]:
        distances = self.assignedPointsDistances[self.branchPointIds[branch]]
        if len(branch.children) == 1:
            distances = np.append(distances, self.getDistances(list(branch.children)[0]))
        return distances


    def getSearchRange(self, branch: Branch) -> float:
        if branch not in self.searchRanges or self.searchRanges[branch] is None:
            distances = self.getDistances(branch)
            if len(distances) > 0:
                self.searchRanges[branch] = np.quantile(distances, 0.9, method="hazen")
            else:
                if branch.isStem():
                    raise ValueError("Expected stem to have assigned points.")
                self.searchRanges[branch] = self.getSearchRange(branch.parent)
        return self.searchRanges[branch]


    def getPointsAtStrip(self,
                         strip: npt.NDArray[np.double],
                         startBranch) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        localIndicesAssigned = self.assignedPointsLookUp.query_ball_point(strip[0], self.getSearchRange(startBranch))

        localIndicesUnassigned = TlsPointManager.findPointIndicesAroundStrip(
            strip, self.unassignedPointsLookUp, self.unassignedPointsDistances, self.unassignedPointsMaxDistance)

        return np.array(localIndicesAssigned, dtype=np.intp), np.array(localIndicesUnassigned, dtype=np.intp)


    def expandBranch(self, branch: Branch):
        if len(branch.children) != 1:
            return

        removedChild = branch.expand()
        branchPointIds = self.branchPointIds[branch]
        childPointIds = self.branchPointIds[removedChild]

        self.searchRanges[branch] = None

        branchId = self.branchIds[branch]
        childId = self.branchIds[removedChild]
        if len(childPointIds) == 0:
            return
        elif len(branchPointIds) == 0: # instead of changing the points, just swap the branches
            self.branches[branchId] = removedChild
            self.branchIds[branch] = childId

            self.branches[childId] = branch
            self.branchIds[removedChild] = branchId
            return

        self.assignedPointsBranchIndices[self.branchPointIds[removedChild]] = branchId
        self.branchPointIds[branch] = np.append(self.branchPointIds[branch], self.branchPointIds[removedChild])
        self.branchPointIds[removedChild] = np.empty(0, dtype=np.intp)


    def splitBranch(self, branch: Branch, splitIdx: int) -> Tuple[Branch, Optional[Branch]]:
        branch, splitEnd = branch.split(splitIdx)

        if splitEnd is not None:
            self.addBranch(splitEnd)
            self.searchRanges[branch] = None

            pointIds = self.branchPointIds[branch]
            tlsPoints = self.assignedPoints[pointIds]
            branchDistances = branch.getDistances(tlsPoints)
            splitEndDistances = splitEnd.getDistances(tlsPoints)

            isAtSplitEnd = splitEndDistances < branchDistances
            self.branchPointIds[splitEnd] = pointIds[isAtSplitEnd]
            self.branchPointIds[branch] = pointIds[~isAtSplitEnd]

            self.assignedPointsBranchIndices[pointIds[isAtSplitEnd]] = len(self.branches) - 1

        return branch, splitEnd


    @staticmethod
    def findPointIndicesAroundStrip(strip: npt.NDArray[np.double],
                                    lookUp: KDTree,
                                    distances: npt.NDArray[np.double],
                                    maxDistance: float
                                    ) -> npt.NDArray[np.intp]:
        stripSegments = np.stack([strip[:-1], strip[1:]], axis=1)

        localIndices = [queryPointsNearSegment(lookUp, s[0], s[1], maxDistance) for s in stripSegments]
        if np.all([len(i) == 0 for i in localIndices]):
            return np.empty(0, dtype=np.intp)

        localMaxDistance = np.max([distances[i].max(initial=0.0) for i in localIndices])
        indices = np.unique(np.concatenate(
            [queryPointsNearSegment(lookUp, s[0], s[1], localMaxDistance) for s in stripSegments], axis=0))
        return indices
