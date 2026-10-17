"""Shared-memory TLS point storage for the multiprocessing connector."""

from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from src.objects.branch import Branch
from src.objects.tlsPointManager import TlsPointManager


class _SharedArray:
    """Pickleable handle for a NumPy array backed by shared memory."""

    def __init__(self, array: Optional[npt.NDArray] = None,
                 name: Optional[str] = None,
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: Optional[str] = None):
        if array is not None:
            self.shape = array.shape
            self.dtype = array.dtype.str
            self.memory = SharedMemory(create=True, size=max(1, array.nbytes))
            self.array = np.ndarray(self.shape, dtype=array.dtype, buffer=self.memory.buf)
            self.array[...] = array
            self.owner = True
        else:
            self.shape = shape
            self.dtype = dtype
            self.memory = SharedMemory(name=name)
            self.array = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=self.memory.buf)
            self.owner = False


    def __getstate__(self):
        return self.memory.name, self.shape, self.dtype


    def __setstate__(self, state):
        name, self.shape, self.dtype = state
        self.memory = SharedMemory(name=name)
        self.array = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=self.memory.buf)
        self.owner = False


    def close(self, unlink: bool = False) -> None:
        self.array = None
        if unlink and self.owner:
            try:
                self.memory.unlink()
            except FileNotFoundError:
                pass
        try:
            self.memory.close()
        finally:
            self.owner = False


class _BranchPointIdsView:
    """Compatibility view exposing a branch's IDs as a NumPy array."""

    def __init__(self, manager: "SharedTlsPointManager"):
        self.manager = manager

    def __getitem__(self, branch: Branch) -> npt.NDArray[np.intp]:
        return self.manager.getBranchPointIds(branch)


class SharedTlsPointManager(TlsPointManager):
    """TLS point manager whose point-sized arrays are shared between processes.

    Branch objects and their slice descriptors remain process-local. The point
    ID buffer, point-sized numeric arrays, and immutable unassigned-point search
    grid live in shared memory. Only the parent process mutates point data;
    workers receive updated slice descriptors after each applied connection.
    """

    _sharedAttributeNames = (
        "assignedPoints", "assignedPointsDistances", "assignedPointsBranchIndices",
        "unassignedPoints", "unassignedPointsDistances", "unassignedPointSearchRadiusGrid", "pointIds"
    )


    def __init__(self, branches: List[Branch], unassignedPoints: npt.NDArray[np.double]):
        super().__init__(branches, unassignedPoints)

        initialBranchPointIds = self.branchPointIds
        totalAssignedPoints = len(self.assignedPoints)
        flatPointIds = np.arange(totalAssignedPoints, dtype=np.intp)
        self._branchPointSlices: Dict[Branch, List[Tuple[int, int]]] = {}
        offset = 0
        for branch in branches:
            pointCount = len(initialBranchPointIds[branch])
            self._branchPointSlices[branch] = [(offset, pointCount)] if pointCount else []
            offset += pointCount

        arrays = {
            "assignedPoints": self.assignedPoints,
            "assignedPointsDistances": self.assignedPointsDistances,
            "assignedPointsBranchIndices": self.assignedPointsBranchIndices.astype(np.intp, copy=False),
            "unassignedPoints": self.unassignedPoints,
            "unassignedPointsDistances": self.unassignedPointsDistances,
            "unassignedPointSearchRadiusGrid": self.unassignedPointSearchRadiusGrid,
            "pointIds": flatPointIds,
        }
        self._sharedArrays = {}
        try:
            for name, array in arrays.items():
                self._sharedArrays[name] = _SharedArray(array)
            for name, sharedArray in self._sharedArrays.items():
                setattr(self, name, sharedArray.array)

            self.branchPointIds = _BranchPointIdsView(self)
            self.assignedPointsLookUp = KDTree(self.assignedPoints)
            self.unassignedPointsLookUp = KDTree(self.unassignedPoints)
        except BaseException:
            for sharedArray in self._sharedArrays.values():
                sharedArray.close(unlink=True)
            raise
        self._sharedWriteEnabled = True
        self._changedBranches = set()


    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._sharedAttributeNames:
            state[name] = None
        state["assignedPointsLookUp"] = None
        state["unassignedPointsLookUp"] = None
        state["_sharedWriteEnabled"] = False
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, sharedArray in self._sharedArrays.items():
            setattr(self, name, sharedArray.array)
        self.assignedPointsLookUp = KDTree(self.assignedPoints)
        self.unassignedPointsLookUp = KDTree(self.unassignedPoints)
        self._sharedWriteEnabled = False


    def closeSharedMemory(self, unlink: bool = False) -> None:
        self.assignedPointsLookUp = None
        self.unassignedPointsLookUp = None
        for name in self._sharedAttributeNames:
            setattr(self, name, None)
        for sharedArray in self._sharedArrays.values():
            sharedArray.close(unlink=unlink)


    def getBranchPointIds(self, branch: Branch) -> npt.NDArray[np.intp]:
        slices = self._branchPointSlices.get(branch, [])
        if not slices:
            return np.empty(0, dtype=np.intp)
        if len(slices) == 1:
            start, length = slices[0]
            return self.pointIds[start:start + length]
        return np.concatenate([self.pointIds[start:start + length] for start, length in slices])


    def exportBranchSlices(self, branchIds: Optional[List[int]] = None) -> Dict[int, List[Tuple[int, int]]]:
        branchIdSet = None if branchIds is None else set(branchIds)
        return {branch.id: list(slices) for branch, slices in self._branchPointSlices.items()
                if branchIdSet is None or branch.id in branchIdSet}


    def applyBranchSlices(self, slicesByBranchId: Dict[int, List[Tuple[int, int]]],
                          changedBranchIds: List[int]) -> None:
        branchesById = {branch.id: branch for branch in self.branches}
        for branchId, slices in slicesByBranchId.items():
            branch = branchesById.get(branchId)
            if branch is not None:
                self._branchPointSlices[branch] = list(slices)
        for branch in self.branches:
            if branch.id in changedBranchIds:
                self.searchRanges[branch] = None


    def resetChangedBranches(self) -> None:
        self._changedBranches.clear()


    def getChangedBranchIds(self) -> List[int]:
        return sorted(branch.id for branch in self._changedBranches)


    def _markChanged(self, branch: Optional[Branch]) -> None:
        if branch is None:
            return
        self._changedBranches.add(branch)
        self.searchRanges[branch] = None


    def _normalizeSlices(self, slices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        normalized = []
        for start, length in sorted(slices):
            if length == 0:
                continue
            if normalized and normalized[-1][0] + normalized[-1][1] == start:
                previousStart, previousLength = normalized[-1]
                normalized[-1] = (previousStart, previousLength + length)
            else:
                normalized.append((start, length))
        return normalized


    def _addBranch(self, branch: Branch) -> int:
        branchId = len(self.branches)
        self.branchIds[branch] = branchId
        self.branches.append(branch)
        self._branchPointSlices[branch] = []
        self._markChanged(branch)
        return branchId


    def addBranch(self, branch: Branch):
        self._addBranch(branch)


    def _partitionSlice(self, start: int, length: int,
                        moveMask: npt.NDArray[np.bool_]
                        ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], npt.NDArray[np.intp]]:
        ids = self.pointIds[start:start + length].copy()
        moveMask = np.asarray(moveMask, dtype=bool)
        movedIds = ids[moveMask]
        keptIds = ids[~moveMask]
        self.pointIds[start:start + len(keptIds)] = keptIds
        self.pointIds[start + len(keptIds):start + length] = movedIds
        keptSlice = (start, len(keptIds)) if len(keptIds) else None
        movedSlice = (start + len(keptIds), len(movedIds)) if len(movedIds) else None
        return keptSlice, movedSlice, movedIds


    def _partitionBranchByOwner(self, sourceBranch: Branch, newBranch: Branch) -> None:
        sourceSlices = self._branchPointSlices.get(sourceBranch, [])
        keptSlices = []
        movedSlices = []
        newBranchSlot = self.branchIds[newBranch]
        for start, length in sourceSlices:
            ids = self.pointIds[start:start + length]
            moveMask = self.assignedPointsBranchIndices[ids] == newBranchSlot
            keptSlice, movedSlice, _ = self._partitionSlice(start, length, moveMask)
            if keptSlice is not None:
                keptSlices.append(keptSlice)
            if movedSlice is not None:
                movedSlices.append(movedSlice)
        self._branchPointSlices[sourceBranch] = self._normalizeSlices(keptSlices)
        self._branchPointSlices[newBranch] = self._normalizeSlices(
            self._branchPointSlices[newBranch] + movedSlices)
        if movedSlices:
            self._markChanged(sourceBranch)


    def createBranch(self,
                     skeletonPoints: npt.NDArray[np.double],
                     parent: Branch,
                     assignedTlsIds: npt.NDArray[np.intp],
                     newAssignedDistances: npt.NDArray[np.double],
                     unassignedTlsIds: npt.NDArray[np.intp],
                     newUnassignedDistances: npt.NDArray[np.double],
                     resolution: float) -> Branch:
        branch = Branch(skeletonPoints, parent=parent)
        Branch.trackEndPoints = True
        branchId = self._addBranch(branch)
        if self._sharedWriteEnabled:
            oldBranchIds = np.unique(self.assignedPointsBranchIndices[assignedTlsIds])
            self.assignedPointsDistances[assignedTlsIds] = newAssignedDistances
            self.assignedPointsBranchIndices[assignedTlsIds] = branchId
            self.unassignedPointsDistances[unassignedTlsIds] = newUnassignedDistances
            for oldBranchId in oldBranchIds:
                oldBranch = self.branches[int(oldBranchId)]
                self._partitionBranchByOwner(oldBranch, branch)
        branch.interpolate(resolution, temporary=True)
        return branch


    def getDistances(self, branch: Branch) -> npt.NDArray[np.double]:
        pointIds = self.getBranchPointIds(branch)
        distances = self.assignedPointsDistances[pointIds]
        if len(branch.children) == 1:
            distances = np.append(distances, self.getDistances(next(iter(branch.children))))
        return distances


    def releasePoints(self) -> npt.NDArray[np.double]:
        unassignedPoints = []
        branches = [branch for branch in self.branches if not branch.isRemoved]
        for branch in branches:
            branch.endPointIndices = None
            branch.trackEndPoints = False
            unassignedPoints.append(Branch.assignLocalPoints(
                self.assignedPoints[self.getBranchPointIds(branch)], [branch], useTrimming=True))
        unassignedPoints.append(Branch.assignAllPoints(self.unassignedPoints, branches, useTrimming=True))
        return np.concatenate(unassignedPoints, axis=0)


    def expandBranch(self, branch: Branch):
        if len(branch.children) != 1:
            return

        removedChild = branch.expand()
        self._markChanged(branch)
        if not self._sharedWriteEnabled:
            return

        branchSlices = self._branchPointSlices.get(branch, [])
        childSlices = self._branchPointSlices.get(removedChild, [])
        branchId = self.branchIds[branch]
        if not childSlices:
            return
        childIds = self.getBranchPointIds(removedChild)
        self.assignedPointsBranchIndices[childIds] = branchId
        self._branchPointSlices[branch] = self._normalizeSlices(branchSlices + childSlices)
        self._branchPointSlices[removedChild] = []
        self._markChanged(removedChild)


    def splitBranch(self, branch: Branch, splitIdx: int) -> Tuple[Branch, Optional[Branch]]:
        branch, splitEnd = branch.split(splitIdx)
        self._markChanged(branch)
        if splitEnd is None:
            return branch, None

        splitEndId = self._addBranch(splitEnd)
        if not self._sharedWriteEnabled:
            return branch, splitEnd

        sourceSlices = self._branchPointSlices.get(branch, [])
        splitSlices = []
        keptSlices = []
        for start, length in sourceSlices:
            pointIds = self.pointIds[start:start + length].copy()
            tlsPoints = self.assignedPoints[pointIds]
            branchDistances = branch.getDistances(tlsPoints)
            splitEndDistances = splitEnd.getDistances(tlsPoints)
            moveMask = splitEndDistances < branchDistances
            keptSlice, movedSlice, movedIds = self._partitionSlice(start, length, moveMask)
            if keptSlice is not None:
                keptSlices.append(keptSlice)
            if movedSlice is not None:
                splitSlices.append(movedSlice)
                self.assignedPointsBranchIndices[movedIds] = splitEndId
        self._branchPointSlices[branch] = self._normalizeSlices(keptSlices)
        self._branchPointSlices[splitEnd] = self._normalizeSlices(splitSlices)
        self._markChanged(branch)
        return branch, splitEnd
