from __future__ import annotations
from itertools import chain
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List, Optional, Tuple

from objects.branch import Branch
from utils.branchUtils import reassignAllPoints
from utils.iteration import breadthFirstSearch
from utils.geometry import angle, normalizeVectors, linesPointsDistances


class Component:
    def __init__(self, branches: List[Branch]):
        self.branches = branches
        self.directionData = {}
        self.boundingBoxMin = np.min(np.stack([b.boundingBoxMin for b in branches]), axis=0)
        self.boundingBoxMax = np.max(np.stack([b.boundingBoxMax for b in branches]), axis=0)


    def getBoundingBoxDistance(self, other: Component | Branch):
        delta1 = np.clip(self.boundingBoxMin - other.boundingBoxMax, a_min=0, a_max=None)
        delta2 = np.clip(other.boundingBoxMin - self.boundingBoxMax, a_min=0, a_max=None)
        return np.linalg.norm(delta1 + delta2, axis=-1)


    def getDirectionData(self,
                         branch: Branch,
                         isEndPoint: npt.NDArray[bool],
                         minDistanceThreshold: float
                         ) -> (Tuple[Optional[npt.NDArray[np.double]], Optional[npt.NDArray[np.double]]] |
                               Tuple[int | npt.NDArray[np.double], Optional[npt.NDArray[np.double]]]):
        if branch not in self.directionData:
            self.generateDirectionData(branch, isEndPoint, minDistanceThreshold)
        return self.directionData[branch]


    def generateDirectionData(self, branch: Branch, isEndPoint: npt.NDArray[bool], minDistanceThreshold: float):
        raise NotImplementedError("only implemented by subclasses")


class ConnectedComponent(Component):
    def __init__(self, branches: List[Branch]):
        Component.__init__(self, branches)


    def extend(self, subComponent: ConnectedComponent):
        self.branches += subComponent.branches
        self.directionData.update(subComponent.directionData)
        self.boundingBoxMin = np.minimum(self.boundingBoxMin, subComponent.boundingBoxMin)
        self.boundingBoxMax = np.maximum(self.boundingBoxMax, subComponent.boundingBoxMax)


    def generateDirectionData(self, branch: Branch, isEndPoint: npt.NDArray[bool], minDistanceThreshold: float):
        self.directionData[branch] = []
        maxIdx = len(branch.skeletonPoints) - 1

        # if the branch has a parent, the first point is handled by the parent branch
        firstHandledIdx = 0 if branch.isStem() else 1 # TODO: rework, since branch is already connected

        for idx in range(firstHandledIdx, maxIdx):
            prePointDirections = getDirections(idx, -1, branch, isEndPoint, minDistanceThreshold)
            postPointDirections = getDirections(idx, 1, branch, isEndPoint, minDistanceThreshold)

            prePointDirections = np.stack(prePointDirections) if len(prePointDirections) > 0 else None
            postPointDirections = np.stack(postPointDirections) if len(postPointDirections) > 0 else None
            self.directionData[branch].append((prePointDirections, postPointDirections))

        # tip point of branch
        prePointDirections = getDirections(maxIdx, -1, branch, isEndPoint, minDistanceThreshold)
        postPointDirections = []
        for child in branch.children:
            postPointDirections.append(getDirections(0, 1, child, isEndPoint, minDistanceThreshold))

        prePointDirections = np.stack(prePointDirections) if len(prePointDirections) > 0 else None
        postPointDirections = np.stack(postPointDirections) if len(postPointDirections) > 0 else None
        self.directionData[branch].append((prePointDirections, postPointDirections))


class UnconnectedComponent(Component):
    def __init__(self, branches: List[Branch]):
        Component.__init__(self, branches)
        self.bestConnection: Optional[Tuple[Branch, np.intp, Branch, np.intp]] = None
        self.bestConnectionScore = 0
        self.directionData = {}

    def connect(self,
                defaultId: np.intp,
                resolution: float
                ) -> ConnectedComponent:
        if self.bestConnection is None:
            raise ValueError("Components can not be connected.")
        startBranch, startIdx, endBranch, endIdx = self.bestConnection
        startPoint = startBranch.skeletonPoints[startIdx]
        endPoint = endBranch.skeletonPoints[endIdx]
        connectionLength = np.linalg.norm(endPoint - startPoint)

        pointNumber = 2 + np.floor(connectionLength / resolution)
        skeletonPoints = np.linspace(startPoint, endPoint, pointNumber, endpoint=True)
        skeletonIDs = np.array([startBranch.skeletonIDs[startIdx]] +
                               (pointNumber - 2) * [defaultId] +
                               [endBranch.skeletonIDs[endIdx]], dtype=np.intp)
        tlsPoints = np.empty((0, 3), dtype=np.double)

        parent = startBranch.split(startIdx)

        connectionBranch = Branch(None, skeletonIDs, skeletonPoints, tlsPoints, parent)

        childBranch = endBranch.split(endIdx)
        if childBranch is None:  # connection end point is root point of the component
            endBranch.setParent(connectionBranch)
        else:
            childBranch.reorientSubtree(connectionBranch)

        connectedComponent = ConnectedComponent(self.branches + [connectionBranch])
        return connectedComponent


    def generateDirectionData(self, branch: Branch, isEndPoint: npt.NDArray[bool], minDistanceThreshold: float):
        self.directionData[branch] = []
        endPointIndices = np.flatnonzero(isEndPoint[branch.skeletonIDs])

        for i, idx in enumerate(endPointIndices):
            if i >= 1 and not directionIsIgnored(idx, endPointIndices[i-1], branch, isEndPoint, minDistanceThreshold):
                startDirection = normalizeVectors(
                    branch.skeletonPoints[idx] - branch.skeletonPoints[endPointIndices[i-1]])
            elif (i+1 < len(endPointIndices) and
                  not directionIsIgnored(idx, endPointIndices[i+1], branch, isEndPoint, minDistanceThreshold)):
                startDirection = normalizeVectors(
                    branch.skeletonPoints[idx] - branch.skeletonPoints[endPointIndices[i+1]])
            else:
                continue
            firstHalfDirections = getDirections(idx, -1, branch, isEndPoint, minDistanceThreshold)
            secondHalfDirections = getDirections(idx, 1, branch, isEndPoint, minDistanceThreshold)
            directions = firstHalfDirections + secondHalfDirections

            directions = np.stack(directions) if len(directions) > 0 else None
            self.directionData[branch].append((idx, startDirection, directions))


def directionIsIgnored(idx_1: int | npt.NDArray[np.intp],
                       idx_2: int | npt.NDArray[np.intp],
                       branch: Branch,
                       isEndPoint: npt.NDArray[bool],
                       minDistanceThreshold: float) -> bool | npt.NDArray[bool]:
    return (isEndPoint[branch.skeletonIDs[idx_1]] & isEndPoint[branch.skeletonIDs[idx_2]] &
            np.linalg.norm(branch.skeletonPoints[idx_1] - branch.skeletonPoints[idx_2]) <= minDistanceThreshold)


def getDirections(idx: int,
                  orientation: int,  # 1 or -1, determines the orientations of directions along the branch
                  branch: Branch,
                  isEndPoint: npt.NDArray[bool],
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
    if not directionIsIgnored(endIdx, idx, branch, isEndPoint, minDistanceThreshold):
        directions.append(normalizeVectors(branch.skeletonPoints[endIdx] - branch.skeletonPoints[idx]))

    nextIdx = endIdx + orientation
    # direction at start of branch
    if (nextIdx != endIdx and
            not directionIsIgnored(nextIdx, idx, branch, isEndPoint, minDistanceThreshold)):
        directions.append(normalizeVectors(branch.skeletonPoints[nextIdx] - branch.skeletonPoints[idx]))

    return directions


def areConnectionAnglesCredible(connectionDirection: npt.NDArray[np.double],
                                parentDirections: Optional[npt.NDArray[np.double]],
                                siblingDirections: Optional[npt.NDArray[np.double]],
                                startDirection: Optional[npt.NDArray[np.double]],
                                childDirections: Optional[npt.NDArray[np.double]],
                                minAngleThreshold: float,
                                maxAngleThreshold: float
                                ) -> bool:
    startToConnectionAngle = angle(startDirection, -connectionDirection)
    if startToConnectionAngle > maxAngleThreshold:
        return False

    if parentDirections is not None:
        anglesToParent = angle(parentDirections, -connectionDirection)
        if np.any(anglesToParent > maxAngleThreshold):
            return False

    if siblingDirections is not None:
        anglesToSiblings = angle(siblingDirections, connectionDirection)
        if np.any(anglesToSiblings <= minAngleThreshold):
            return False

    if childDirections is not None:
        anglesToChildren = angle(childDirections, connectionDirection)
        if np.any(anglesToChildren > maxAngleThreshold):
            return False
        revAnglesToChildren = angle(childDirections, -connectionDirection)
        if np.any(revAnglesToChildren <= minAngleThreshold):
            return False

    return True


def isConnectionDensityCredible(startPoint: npt.NDArray[np.double],
                                endPoint: npt.NDArray[np.double],
                                pedalPoints: npt.NDArray[np.double],
                                windowSize: float,
                                densityFactor: float) -> bool:
    vector = endPoint - startPoint
    length = np.linalg.norm(vector)
    pointsPerWindow = len(pedalPoints) / length * densityFactor * windowSize

    relativeWindowSize = windowSize / length
    mainAxis = vector.argmax()

    projectedPedalPoints = pedalPoints[mainAxis]
    maxProjectedWindowSize = relativeWindowSize * vector[mainAxis]

    projectedPedalPoints = np.append(projectedPedalPoints, (startPoint[mainAxis], endPoint[mainAxis]), axis=0)
    projectedPedalPoints.sort()

    projectedWindowSizes = projectedPedalPoints[pointsPerWindow:] - projectedPedalPoints[:-pointsPerWindow]

    isCredible = np.all(projectedWindowSizes <= maxProjectedWindowSize)
    return isCredible


def getComponentPoints(component: Component,
                       tlsPointLookUp: KDTree
                       ) -> Tuple[npt.NDArray[np.double], List[npt.NDArray[np.double]]]:
    componentPointIndices, searchRanges = [], []
    distances = [branch.getDistances(branch.tlsPoints) for branch in component.branches]
    allDistances = np.concatenate(distances)
    defaultSearchRange = np.quantile(allDistances, 0.9, axis=0, method='hazen')

    for branch, distance in zip(component.branches, distances):
        searchRange = np.quantile(distance, 0.9, axis=0, method='hazen') if len(distance) > 0 else defaultSearchRange
        searchRanges.append(searchRange)

        componentPointIndices.extend(tlsPointLookUp.query_ball_point(branch.skeletonPoints, searchRange))

    componentPointIndices = np.array(componentPointIndices).flatten()
    componentPointIndices = np.unique(componentPointIndices)
    componentPoints = tlsPointLookUp.data[componentPointIndices]
    return componentPoints, searchRanges



def connectComponents(rootComponentStem: Branch,
                      unconnectedComponentStems: List[Branch],
                      unassignedPoints: npt.NDArray[np.double],
                      isEndPoint: npt.NDArray[bool],
                      minDistanceThreshold: float,
                      maxDistanceThreshold: float,
                      minAngleThreshold: float,
                      maxAngleThreshold: float,
                      resolution: float,
                      windowSize: float,
                      densityFactor: float):
    rootComponent = ConnectedComponent(list(breadthFirstSearch(rootComponentStem)))
    unconnectedComponents = [UnconnectedComponent(list(breadthFirstSearch(stem)))
                             for stem in unconnectedComponentStems if stem is not rootComponentStem]

    originalComponentCount = unconnectedComponents + 1

    allBranches = rootComponent.branches + list(chain.from_iterable([c.branches for c in unconnectedComponents]))

    assignedPoints = np.concatenate([b.tlsPoints for b in allBranches], axis=0)
    assignedPointsLookUp = KDTree(assignedPoints)

    defaultId = np.flatnonzero(~isEndPoint)[0]

    unassignedPointsLookUp = KDTree(unassignedPoints)

    originalMaxDistanceThreshold = maxDistanceThreshold
    maxCheckedDistance = 0.0
    currentComponent: ConnectedComponent = rootComponent

    while unconnectedComponents:
        connectionComponent = None

        for otherComponent in unconnectedComponents:
            if currentComponent.getBoundingBoxDistance(otherComponent) <= maxDistanceThreshold:
                updateBestConnection(currentComponent, otherComponent, allBranches,
                                     assignedPointsLookUp, unassignedPointsLookUp, isEndPoint,
                                     minDistanceThreshold, maxDistanceThreshold, maxCheckedDistance,
                                     minAngleThreshold, maxAngleThreshold,
                                     windowSize, densityFactor)

            if otherComponent.bestConnectionScore > bestScore:
                bestScore = otherComponent.bestConnectionScore
                connectionComponent = otherComponent

        if currentComponent is not rootComponent:
            rootComponent.extend(currentComponent)

        if connectionComponent is not None:
            del(unconnectedComponents[unconnectedComponents.index(connectionComponent)])

            currentComponent = connectionComponent.connect(defaultId, resolution)
            maxDistanceThreshold = originalMaxDistanceThreshold
            maxCheckedDistance = 0.0
        else:
            if maxDistanceThreshold >= 2.0 or len(unconnectedComponents) <= originalComponentCount / 10:
                break
            currentComponent = rootComponent
            maxCheckedDistance = maxDistanceThreshold
            maxDistanceThreshold += 0.2

    for branch in rootComponent.branches:
        if len(branch.children) == 1:
            branch.expand()

    newUnassignedPoints1 = reassignAllPoints(rootComponent.branches, useTrimming=True)
    allBranches = rootComponent.branches + list(chain.from_iterable([c.branches for c in unconnectedComponents]))
    newUnassignedPoints2 = Branch.assignAllPoints(allBranches, unassignedPoints, useTrimming=True)
    unassignedPoints = np.concatenate((newUnassignedPoints1, newUnassignedPoints2), axis=0)

    for branch in rootComponent.branches:
        if branch.isStem():
            return branch, unassignedPoints
    raise ValueError("No stem was found")


def updateBestConnection(rootComponent: ConnectedComponent,
                         unconnectedComponent: UnconnectedComponent,
                         allBranches: List[Branch],
                         assignedPointsLookUp: KDTree,
                         unassignedPointsLookUp: KDTree,
                         isEndPoint: npt.NDArray[np.bool],
                         minDistanceThreshold: float,
                         maxDistanceThreshold: float,
                         maxCheckedDistance: float,
                         minAngleThreshold: float,
                         maxAngleThreshold: float,
                         windowSize: float,
                         densityFactor: float):
    rootPoints, searchRanges = getComponentPoints(rootComponent, assignedPointsLookUp)
    rootPointsLookUp = KDTree(rootPoints)
    _, rootPointDistances = Branch.findClosestBranches(rootPoints, rootComponent.branches)

    for endBranch in unconnectedComponent.branches:
        if rootComponent.getBoundingBoxDistance(endBranch) > maxDistanceThreshold:
            continue
        directionsAtEnd = unconnectedComponent.getDirectionData(endBranch, isEndPoint, minDistanceThreshold)

        for startBranch, searchRange in zip(rootComponent.branches, searchRanges):
            if startBranch.branchBoundingBoxDistance(endBranch) > maxDistanceThreshold:
                continue
            directionsAtStart = rootComponent.getDirectionData(startBranch, isEndPoint, minDistanceThreshold)

            for endIdx, startDirection, childDirections in directionsAtEnd:
                for startIdx, (parentDirections, siblingDirections) in enumerate(directionsAtStart):
                    if len(directionsAtStart) < len(startBranch.skeletonPoints):
                        startIdx += 1

                    connectionLength = np.linalg.norm(
                        startBranch.skeletonPoints[startIdx] - endBranch.skeletonPoints[endIdx])

                    if connectionLength > maxDistanceThreshold or connectionLength <= maxCheckedDistance:
                        continue

                    score = scoreConnection(startBranch, startIdx, endBranch, endIdx, searchRange,
                                            startDirection, childDirections, parentDirections, siblingDirections,
                                            allBranches, rootPointsLookUp, unassignedPointsLookUp,
                                            minAngleThreshold, maxAngleThreshold, windowSize, densityFactor)
                    if score > unconnectedComponent.bestConnectionScore:
                        unconnectedComponent.bestConnection = (startBranch, startIdx, endBranch, endIdx)
                        unconnectedComponent.bestConnectionScore = score


def scoreConnection(startBranch: Branch,
                    startIdx: int,
                    endBranch: Branch,
                    endIdx: int,
                    searchRange: float,
                    startDirection: npt.NDArray[np.double],
                    childDirections: npt.NDArray[np.double],
                    parentDirections: npt.NDArray[np.double],
                    siblingDirections: npt.NDArray[np.double],
                    allBranches: List[Branch],
                    rootPointsLookUp: KDTree,
                    unassignedPointsLookUp: KDTree,
                    minAngleThreshold: float,
                    maxAngleThreshold: float,
                    windowSize: float,
                    densityFactor: float
                    ) -> float:
    startPoint = startBranch.skeletonPoints[startIdx]
    endPoint = endBranch.skeletonPoints[endIdx]
    connectionVector = endPoint - startPoint
    connectionDirection = normalizeVectors(connectionVector)
    connectionLength = np.linalg.norm(connectionVector)

    if not areConnectionAnglesCredible(connectionDirection,
                                       parentDirections, siblingDirections, startDirection, childDirections,
                                       minAngleThreshold, maxAngleThreshold):
        return 0

    midPoint = startPoint + 0.5*connectionVector
    searchRadius = 0.5*connectionLength + 2*searchRange
    closeRootPoints = rootPointsLookUp.query_ball_point(midPoint, searchRadius)
    closeUnassignedPoints = unassignedPointsLookUp.query_ball_point(midPoint, searchRadius)
    closePoints = np.concatenate((closeRootPoints, closeUnassignedPoints), axis=0)

    _, branchDistances = Branch.findLocalClosestBranches(closePoints, allBranches,
                                                         [startBranch, endBranch])

    pedalDistances, pedalPoints = linesPointsDistances(startPoint.reshape(1, 3),
                                                       endPoint.reshape(1, 3),  # TODO: test/refactor
                                                       closePoints,
                                                       returnPedalPoints=True)

    pointIsAtConnection = pedalDistances < branchDistances & pedalDistances < 2 * searchRange
    connectionPoints = closePoints[pointIsAtConnection]
    pedalPoints = pedalPoints[pointIsAtConnection]
    pedalDistances = pedalDistances[pointIsAtConnection]

    if (len(connectionPoints) == 0 or
            not isConnectionDensityCredible(startPoint, endPoint, pedalPoints, windowSize, densityFactor)):
        return 0

    score = len(connectionPoints) / connectionLength / np.mean(pedalDistances)

    return score
