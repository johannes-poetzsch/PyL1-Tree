from itertools import product
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List, Optional, Set, Tuple

from src.objects.branch import Branch
from src.utils.geometry import queryPointsNearStrip


class TlsPointManager:
    unassignedPointGridCellSize: float = 0.5

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
        self.unassignedPointsMaxDistance = np.max(self.unassignedPointsDistances, initial=0.0)
        # These bounds use the initial distances and remain conservative as distances decrease.
        self.unassignedPointGridOrigin, self.unassignedPointSearchRadiusGrid = (
            self._createUnassignedPointSearchRadiusGrid(branches))

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
        localIndicesUnassigned = self.getUnassignedPointIndicesAroundStrip(strip)

        return np.array(localIndicesAssigned, dtype=np.intp), np.array(localIndicesUnassigned, dtype=np.intp)


    def getUnassignedPointIndicesAroundStrip(self, strip: npt.NDArray[np.double]) -> npt.NDArray[np.intp]:
        searchRadius = self._getUnassignedPointSearchRadius(strip)
        if searchRadius <= 0:
            return np.empty(0, dtype=np.intp)

        indices = queryPointsNearStrip(self.unassignedPointsLookUp, strip, searchRadius)
        return np.unique(indices)


    def _getUnassignedPointSearchRadius(self, strip: npt.NDArray[np.double]) -> float:
        cells = self._getGridCellsForStrip(strip)
        if cells is None:
            # Candidate strips normally stay inside the original skeleton bounds.
            # Retain a safe fallback if an unusual input leaves the indexed area.
            return self.unassignedPointsMaxDistance
        if not cells:
            return 0.0

        return max(self.unassignedPointSearchRadiusGrid[cell] for cell in cells)


    def _getGridCellsForStrip(self, strip: npt.NDArray[np.double]) -> Optional[Set[Tuple[int, int, int]]]:
        if len(strip) == 0:
            return set()

        cellSize = self.unassignedPointGridCellSize
        gridShape = self.unassignedPointSearchRadiusGrid.shape
        cells: Set[Tuple[int, int, int]] = set()
        segments = zip(strip[:-1], strip[1:]) if len(strip) > 1 else [(strip[0], strip[0])]

        for startPoint, endPoint in segments:
            segmentMin = np.minimum(startPoint, endPoint)
            segmentMax = np.maximum(startPoint, endPoint)
            minCell = np.floor((segmentMin - self.unassignedPointGridOrigin) / cellSize).astype(int) - 1
            maxCell = np.floor((segmentMax - self.unassignedPointGridOrigin) / cellSize).astype(int)

            for x in range(minCell[0], maxCell[0] + 1):
                for y in range(minCell[1], maxCell[1] + 1):
                    for z in range(minCell[2], maxCell[2] + 1):
                        cell = (x, y, z)
                        cellMin = self.unassignedPointGridOrigin + np.asarray(cell) * cellSize
                        cellMax = cellMin + cellSize
                        if not TlsPointManager._segmentIntersectsCell(startPoint, endPoint, cellMin, cellMax):
                            continue
                        if any(index < 0 or index >= gridShape[axis] for axis, index in enumerate(cell)):
                            return None
                        cells.add(cell)

        return cells


    @staticmethod
    def _segmentIntersectsCell(startPoint: npt.NDArray[np.double],
                               endPoint: npt.NDArray[np.double],
                               cellMin: npt.NDArray[np.double],
                               cellMax: npt.NDArray[np.double]) -> bool:
        direction = endPoint - startPoint
        minParameter, maxParameter = 0.0, 1.0
        for axis in range(3):
            if direction[axis] == 0:
                if startPoint[axis] < cellMin[axis] or startPoint[axis] > cellMax[axis]:
                    return False
                continue

            first = (cellMin[axis] - startPoint[axis]) / direction[axis]
            second = (cellMax[axis] - startPoint[axis]) / direction[axis]
            minParameter = max(minParameter, min(first, second))
            maxParameter = min(maxParameter, max(first, second))
            if minParameter > maxParameter:
                return False

        return True


    def _createUnassignedPointSearchRadiusGrid(self,
                                               branches: List[Branch]
                                              ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        """Index an upper bound for every point that could improve a strip in each cell."""
        cellSize = self.unassignedPointGridCellSize
        coordinates = [branch.skeletonPoints for branch in branches]
        if len(self.unassignedPoints) > 0:
            coordinates.append(self.unassignedPoints)
        allCoordinates = np.concatenate(coordinates, axis=0)

        firstCell = np.floor(allCoordinates.min(axis=0) / cellSize).astype(int) - 1
        lastCell = np.floor(allCoordinates.max(axis=0) / cellSize).astype(int) + 1
        gridOrigin = firstCell.astype(np.double) * cellSize
        gridShape = tuple((lastCell - firstCell + 1).astype(int))
        searchRadiusGrid = np.zeros(gridShape, dtype=np.double)

        points = self.unassignedPoints
        radii = self.unassignedPointsDistances
        hasInfluence = radii > 0
        # A point only affects cells within ceil(radius / cellSize) cells of its own cell.
        # Points with the same reach are processed together, one cell offset at a time.
        levels = np.ceil(radii / cellSize).astype(np.intp)
        flatGrid = searchRadiusGrid.reshape(-1)
        gridShapeArray = np.asarray(gridShape)
        gridY, gridZ = gridShape[1], gridShape[2]
        for level in np.unique(levels[hasInfluence]):
            isLevel = hasInfluence & (levels == level)
            levelPoints = points[isLevel]
            levelRadii = radii[isLevel]
            pointCells = np.floor((levelPoints - gridOrigin) / cellSize).astype(np.intp)
            for offset in product(range(-level, level + 1), repeat=3):
                cellIndices = pointCells + np.array(offset, dtype=np.intp)
                isInGrid = np.all((cellIndices >= 0) & (cellIndices < gridShapeArray), axis=1)
                if not np.any(isInGrid):
                    continue

                selectedPoints = levelPoints[isInGrid]
                selectedRadii = levelRadii[isInGrid]
                selectedCells = cellIndices[isInGrid]
                cellMin = gridOrigin + selectedCells * cellSize
                cellMax = cellMin + cellSize
                minDelta = np.maximum(np.maximum(cellMin - selectedPoints, 0.0), selectedPoints - cellMax)
                intersectsInfluence = np.vecdot(minDelta, minDelta, axis=1) <= selectedRadii ** 2
                if not np.any(intersectsInfluence):
                    continue

                cellMin = cellMin[intersectsInfluence]
                cellMax = cellMax[intersectsInfluence]
                selectedPoints = selectedPoints[intersectsInfluence]
                maxDelta = np.maximum(np.abs(cellMin - selectedPoints), np.abs(cellMax - selectedPoints))
                cellSearchRadii = np.sqrt(np.vecdot(maxDelta, maxDelta, axis=1))
                selectedCells = selectedCells[intersectsInfluence]
                linearIndices = (selectedCells[:, 0] * gridY + selectedCells[:, 1]) * gridZ + selectedCells[:, 2]
                np.maximum.at(flatGrid, linearIndices, np.minimum(cellSearchRadii, self.unassignedPointsMaxDistance))

        return gridOrigin, searchRadiusGrid


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
        localIndices = queryPointsNearStrip(lookUp, strip, maxDistance)
        if len(localIndices) == 0:
            return np.empty(0, dtype=np.intp)

        localMaxDistance = np.max(distances[localIndices], initial=0.0)
        indices = queryPointsNearStrip(lookUp, strip, localMaxDistance)
        return np.unique(indices)
