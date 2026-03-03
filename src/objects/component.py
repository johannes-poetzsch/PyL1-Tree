from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Iterable, List, Optional, Set, Tuple

from src.objects.branch import Branch
from src.utils.geometry import angle, normalizeVectors


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


    def getPoints(self, tlsPointLookUp: KDTree
                  ) -> Tuple[npt.NDArray[np.double], List[npt.NDArray[np.double]]]:
        componentPointIndices, searchRanges = [], []
        distances = [branch.getDistances(branch.tlsPoints) for branch in self.branches]
        allDistances = np.concatenate(distances)
        defaultSearchRange = np.quantile(allDistances, 0.9, axis=0, method='hazen')

        for branch, distance in zip(self.branches, distances):
            searchRange = np.quantile(distance, 0.9, axis=0, method='hazen') if len(distance) > 0 else defaultSearchRange
            searchRanges.append(searchRange)
            pointIndices = tlsPointLookUp.query_ball_point(branch.skeletonPoints, searchRange)
            componentPointIndices.extend([np.array(indices, dtype=np.intp) for indices in pointIndices])

        componentPointIndices = np.concatenate(componentPointIndices)
        componentPointIndices = np.unique(componentPointIndices)
        componentPoints = tlsPointLookUp.data[componentPointIndices]
        return componentPoints, searchRanges


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        raise NotImplementedError("only implemented by subclasses")


    @staticmethod
    def directionIsIgnored(idx_1: int | npt.NDArray[np.intp],
                           idx_2: int | npt.NDArray[np.intp],
                           branch: Branch,
                           minDistanceThreshold: float) -> bool | npt.NDArray[bool]:
        return (branch.isEndPoint(idx_1) & branch.isEndPoint(idx_2) &
                (np.linalg.norm(branch.skeletonPoints[idx_1] - branch.skeletonPoints[idx_2]) <= minDistanceThreshold))


    @staticmethod
    def getDirections(idx: int,
                      orientation: int,  # 1 or -1, determines the orientations of directions along the branch
                      branch: Branch,
                      minDistanceThreshold: float) -> List[npt.NDArray[np.double]]:
        if orientation == 1:
            endIdx = len(branch.skeletonPoints) - 1
        elif orientation == -1:
            endIdx = 0
        else:
            raise ValueError("Orientation must be 1 or -1")

        if idx == endIdx:
            return []

        directions = []
        # direction of whole branch
        if not Component.directionIsIgnored(endIdx, idx, branch, minDistanceThreshold):
            directions.append(normalizeVectors(branch.skeletonPoints[endIdx] - branch.skeletonPoints[idx]))

        nextIdx = endIdx + orientation
        # direction at start of branch
        if (nextIdx < endIdx and
                not Component.directionIsIgnored(nextIdx, idx, branch, minDistanceThreshold)):
            directions.append(normalizeVectors(branch.skeletonPoints[nextIdx] - branch.skeletonPoints[idx]))

        return directions


class ConnectedComponent(Component):
    def __init__(self, branches: List[Branch], pointsLookUp: KDTree):
        Component.__init__(self, branches)
        self.checkedDistanceThreshold = 0.0
        tlsPoints, self.searchRanges = self.getPoints(pointsLookUp)
        self.tlsPointsLookUp = KDTree(tlsPoints)


    def extend(self, subComponent: ConnectedComponent):
        self.branches += subComponent.branches
        self.directionData.update(subComponent.directionData)
        self.boundingBoxMin = np.minimum(self.boundingBoxMin, subComponent.boundingBoxMin)
        self.boundingBoxMax = np.maximum(self.boundingBoxMax, subComponent.boundingBoxMax)


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        self.directionData[branch] = []
        maxIdx = len(branch.skeletonPoints) - 1

        # if the branch has a parent, the first point is handled by the parent branch
        firstHandledIdx = 0 if branch.isStem() else 1 # TODO: rework, since branch is already connected

        for idx in range(firstHandledIdx, maxIdx):
            prePointDirections = Component.getDirections(idx, -1, branch, minDistanceThreshold)
            postPointDirections = Component.getDirections(idx, 1, branch, minDistanceThreshold)

            prePointDirections = np.stack(prePointDirections) if len(prePointDirections) > 0 else None
            postPointDirections = np.stack(postPointDirections) if len(postPointDirections) > 0 else None
            self.directionData[branch].append((prePointDirections, postPointDirections))

        # tip point of branch
        prePointDirections = Component.getDirections(maxIdx, -1, branch, minDistanceThreshold)
        postPointDirections = []
        for child in branch.children:
            postPointDirections.extend(Component.getDirections(0, 1, child, minDistanceThreshold))

        prePointDirections = np.stack(prePointDirections) if len(prePointDirections) > 0 else None
        postPointDirections = np.stack(postPointDirections) if len(postPointDirections) > 0 else None
        self.directionData[branch].append((prePointDirections, postPointDirections))


class UnconnectedComponent(Component):
    def __init__(self, branches: List[Branch]):
        Component.__init__(self, branches)
        self.possibleConnections: List[Tuple[float, Branch, int, Branch, int]] = []
        self.directionData = {}


    def connect(self, resolution: float, pointsLookUp: KDTree) -> Tuple[ConnectedComponent, Branch, Optional[Branch]]:
        if len(self.possibleConnections) == 0:
            raise ValueError("Components can not be connected.")
        score, startBranch, startIdx, endBranch, endIdx = self.possibleConnections[0]
        startPoint = startBranch.skeletonPoints[startIdx]
        endPoint = endBranch.skeletonPoints[endIdx]
        connectionLength = np.linalg.norm(endPoint - startPoint)

        pointNumber = int(2 + np.floor(connectionLength / resolution))
        skeletonPoints = np.linspace(startPoint, endPoint, pointNumber, endpoint=True)
        tlsPoints = np.empty((0, 3), dtype=np.double)

        parent, sibling = startBranch.split(startIdx) # TODO: Save splitEnd in branches

        connectionBranch = Branch(skeletonPoints, tlsPoints, parent)

        childBranch, _ = endBranch.split(endIdx)
        if childBranch is None:  # connection end point is root point of the component
            endBranch.setParent(connectionBranch)
        else:
            childBranch.reorientSubtree(connectionBranch)

        connectedComponent = ConnectedComponent(self.branches + [connectionBranch], pointsLookUp)

        return connectedComponent, connectionBranch, sibling


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        self.directionData[branch] = []
        # TODO: refactor lookup
        endPointIndices = np.flatnonzero(branch.isEndPoint(np.arange(len(branch.skeletonPoints), dtype=int)))

        for i, idx in enumerate(endPointIndices):
            if i >= 1 and not self.directionIsIgnored(idx, endPointIndices[i-1], branch, minDistanceThreshold):
                startDirection = normalizeVectors(
                    branch.skeletonPoints[idx] - branch.skeletonPoints[endPointIndices[i-1]])
            elif (i+1 < len(endPointIndices) and
                  not self.directionIsIgnored(idx, endPointIndices[i+1], branch, minDistanceThreshold)):
                startDirection = normalizeVectors(
                    branch.skeletonPoints[idx] - branch.skeletonPoints[endPointIndices[i+1]])
            else:
                continue
            firstHalfDirections = self.getDirections(idx, -1, branch, minDistanceThreshold)
            secondHalfDirections = self.getDirections(idx, 1, branch, minDistanceThreshold)
            directions = firstHalfDirections + secondHalfDirections

            directions = np.stack(directions) if len(directions) > 0 else None
            self.directionData[branch].append((idx, startDirection, directions))


    def updatePossibleConnections(self,
                                  newBranch: Branch,
                                  minAngleThreshold: float,
                                  minDistanceThreshold: float):
        def updateConnection(connection: Tuple[float, Branch, int, Branch, int]
                             ) -> Tuple[float, Branch, int, Branch, int]:
            score, startBranch, startIdx, endBranch, endIdx = connection
            startBranchLength = len(startBranch.skeletonPoints)
            if newBranch.parent is not startBranch or startIdx < startBranchLength:
                return connection
            else:
                for newStartBranch in startBranch.children:
                    if newStartBranch is not newBranch:
                        return score, newStartBranch, startIdx - startBranchLength + 1, endBranch, endIdx
                raise ValueError("Connection between branch and startBranch not found.")

        def connectionStaysCredible(connection: Tuple[float, Branch, int, Branch, int]) -> bool:
            _, startBranch, startIdx, endBranch, endIdx = connection
            if newBranch.parent is not startBranch or startIdx != len(startBranch.skeletonPoints) - 1:
                return True
            connectionDirection = normalizeVectors(endBranch.skeletonPoints[endIdx] - startBranch.skeletonPoints[startIdx])
            newSiblingDirections = self.getDirections(0, 1, newBranch, minDistanceThreshold)
            return angle(connectionDirection, newSiblingDirections) >= minAngleThreshold

        self.possibleConnections = [updateConnection(c) for c in self.possibleConnections if connectionStaysCredible(c)]
