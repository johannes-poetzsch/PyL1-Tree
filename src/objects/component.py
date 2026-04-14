from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Dict, List, Optional, Tuple

from src.objects.branch import Branch
from src.scoreConnection import areConnectionAnglesCredible
from src.utils.geometry import normalizeVectors


class Component:
    def __init__(self, branches: List[Branch]):
        self.branches = branches
        self.directionData = {}
        self.boundingBoxMin = np.min(np.stack([b.boundingBoxMin for b in branches]), axis=0)
        self.boundingBoxMax = np.max(np.stack([b.boundingBoxMax for b in branches]), axis=0)


    def getBoundingBoxDistance(self, other: Component | Branch | npt.NDArray[np.double]) -> float:
        if isinstance(other, Branch) or isinstance(other, Component):
            otherMax = other.boundingBoxMax
            otherMin = other.boundingBoxMin
        else:
            otherMax = other
            otherMin = other
        delta1 = np.clip(self.boundingBoxMin - otherMax, a_min=0, a_max=None)
        delta2 = np.clip(otherMin - self.boundingBoxMax, a_min=0, a_max=None)
        return np.linalg.norm(delta1 + delta2, axis=-1)


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


class ConnectedComponent(Component):
    def __init__(self, branches: List[Branch], tlsLookUp: KDTree, tlsDistances: npt.NDArray[np.double]):
        Component.__init__(self, branches)
        self.checkedDistanceThreshold = 0.0
        self.tlsLookUp = tlsLookUp
        self.tlsDistances = tlsDistances
        self.searchRanges = self.generateSearchRanges()


    def adjustSplitBranch(self, splitEnd: Optional[Branch]):
        if splitEnd is not None:
            self.branches.append(splitEnd)
            self.searchRanges[splitEnd] = self.searchRanges[splitEnd.parent]


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        if branch.id == 78:
            pass
        self.directionData[branch] = []
        maxIdx = len(branch) - 1

        # if the branch has a parent, the first point is handled by the parent branch
        firstHandledIdx = 0 if branch.isStem() else 1 # TODO: rework, since branch is already connected

        for idx in range(firstHandledIdx, maxIdx):
            if idx == 221:
                pass
            prePointDirections = Component.getDirections(idx, -1, branch, minDistanceThreshold)
            postPointDirections = Component.getDirections(idx, 1, branch, minDistanceThreshold)

            prePointDirections = np.stack(prePointDirections) if len(prePointDirections) > 0 else None
            postPointDirections = np.stack(postPointDirections) if len(postPointDirections) > 0 else None
            self.directionData[branch].append((idx, prePointDirections, postPointDirections))

        # tip point of branch
        prePointDirections = Component.getDirections(maxIdx, -1, branch, minDistanceThreshold)
        postPointDirections = []
        for child in branch.children:
            postPointDirections.extend(Component.getDirections(0, 1, child, minDistanceThreshold))

        prePointDirections = np.stack(prePointDirections) if len(prePointDirections) > 0 else None
        postPointDirections = np.stack(postPointDirections) if len(postPointDirections) > 0 else None
        self.directionData[branch].append((maxIdx, prePointDirections, postPointDirections))


    def generateSearchRanges(self) -> Dict[Branch, float]:
        distances = [branch.getDistances(branch.tlsPoints) for branch in self.branches]
        allDistances = np.concatenate(distances)
        defaultSearchRange = np.quantile(allDistances, 0.9, axis=0, method='hazen')

        searchRanges = {b: (np.quantile(d, 0.9, axis=0, method='hazen') if len(d) > 0 else defaultSearchRange)
                        for b, d in zip(self.branches, distances)}

        return searchRanges


    def getTlsData(self, branch: Branch, idx: int) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        searchRange = self.searchRanges[branch]
        indices = self.tlsLookUp.query_ball_point(branch[idx], searchRange)
        tlsPoints = self.tlsLookUp.data[indices]
        distances = self.tlsDistances[indices]
        return tlsPoints, distances


class UnconnectedComponent(Component):
    def __init__(self, branches: List[Branch]):
        Component.__init__(self, branches)
        self.connections: List[Tuple[float, Branch, int, Branch, int, npt.NDArray[np.double], ConnectedComponent]] = []
        self.directionData = {}


    def connect(self, tlsLookUp: KDTree, tlsDistances: npt.NDArray[np.double]
                ) -> Tuple[ConnectedComponent, Branch, Optional[Branch]]:
        if len(self.connections) == 0:
            raise ValueError("Components can not be connected.")
        score, startBranch, startIdx, endBranch, endIdx, connectionStrip, component = self.connections[0]

        parent, sibling = startBranch.split(startIdx)
        component.adjustSplitBranch(sibling)

        connectionBranch = Branch(connectionStrip, parent=parent)

        misorientedChild, newChild = endBranch.split(endIdx)
        if misorientedChild is not None:
            misorientedChild.reorientSubtree(connectionBranch)

        newBranches = [connectionBranch] if newChild is None else [connectionBranch, newChild]
        connectedComponent = ConnectedComponent(self.branches + newBranches, tlsLookUp, tlsDistances)

        return connectedComponent, connectionBranch, sibling


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        if branch.id == 106:
            pass
        self.directionData[branch] = []
        connectionStarts = []

        for startIdx, endIdx in UnconnectedComponent.getBranchSegments(branch):
            startDirection = normalizeVectors(branch[startIdx] - branch[endIdx])
            connectionStarts.append((startIdx, startDirection))
            connectionStarts.append((endIdx, -startDirection))

        for idx, startDirection in connectionStarts:
            firstHalfDirections = self.getDirections(idx, -1, branch, minDistanceThreshold)
            secondHalfDirections = self.getDirections(idx, 1, branch, minDistanceThreshold)
            directions = firstHalfDirections + secondHalfDirections

            directions = np.stack(directions) if len(directions) > 0 else None
            self.directionData[branch].append((idx, startDirection, directions))


    def updateConnections(self,
                          newBranch: Branch,
                          minAngleThreshold: float,
                          minDistanceThreshold: float):
        def updateConnection(connection: Tuple[float, Branch, int, Branch, int, npt.NDArray[np.double], ConnectedComponent]
                             ) -> Tuple[float, Branch, int, Branch, int, npt.NDArray[np.double], ConnectedComponent]:
            score, startBranch, startIdx, endBranch, endIdx, points, component = connection
            if newBranch.parent is not startBranch or startIdx < len(startBranch):
                return connection
            else:
                for newStartBranch in startBranch.children:
                    if newStartBranch is not newBranch:
                        return score, newStartBranch, startIdx - len(startBranch) + 1, endBranch, endIdx, points, component
                raise ValueError("Connection between branch and startBranch not found.")

        def connectionStaysCredible(connection: Tuple[float, Branch, int, Branch, int, npt.NDArray[np.double], ConnectedComponent]
                                    ) -> bool:
            _, startBranch, startIdx, endBranch, endIdx, points, _ = connection
            if newBranch.parent is not startBranch or startIdx != len(startBranch) - 1:
                return True
            connectionDirectionsAtStart = [normalizeVectors(points[-1] - points[0])]
            if len(points) > 2:
                connectionDirectionsAtStart.append(normalizeVectors(points[1] - points[0]))
            newSiblingDirections = self.getDirections(0, 1, newBranch, minDistanceThreshold)
            return areConnectionAnglesCredible(np.array(connectionDirectionsAtStart), np.array([], dtype=np.double),
                                               siblingDirections=np.array(newSiblingDirections),
                                               minAngleThreshold=minAngleThreshold)

        self.connections = [updateConnection(c) for c in self.connections if connectionStaysCredible(c)]


    # Each branch in a component was created by fusing branch segments at their end points.
    # IsEndPoint returns whether a skeleton point in a branch was such an end point.
    # The startDirection is the direction of the branch segment starting from an end point.
    # Since branches share points with their parent, the first (end) point in a branch can either belong
    # to the first branch segment of that branch or to the last branch segment of its parent.
    # Depending on this, the first branch segment either starts at the first or second branch point.
    @staticmethod
    def getBranchSegments(branch: Branch):
        isEndPoint = branch.isEndPoint(np.arange(len(branch), dtype=int))
        endPointIndices = np.flatnonzero(isEndPoint)

        if branch.isStem(): # stems must start with a branch segment
            firstStart = 0
        elif branch.isLeaf(): # leafs must end with a branch segment
            firstStart = len(endPointIndices) % 2
        elif len(endPointIndices) < len(branch):
            nonEndPointIndices = np.flatnonzero(np.logical_not(isEndPoint))
            firstStart = 1 - nonEndPointIndices[0] % 2
        else:
            parentSegments = UnconnectedComponent.getBranchSegments(branch.parent)
            parentEndsInSegment = parentSegments[-1][1] == len(branch.parent) - 1
            firstStart = 1 if parentEndsInSegment else 0

        segments = [(endPointIndices[i], endPointIndices[i+1]) for i in range(firstStart, len(endPointIndices)-1, 2)]
        return segments