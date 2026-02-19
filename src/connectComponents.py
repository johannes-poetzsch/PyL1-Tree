from itertools import chain
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List, Optional, Tuple

from objects.branch import Branch
from objects.component import ConnectedComponent, UnconnectedComponent
from utils.iteration import breadthFirstSearch
from utils.geometry import angle, normalizeVectors, stripPointsDistances


def connectComponents(rootComponentStem: Branch,
                      unconnectedComponentStems: List[Branch],
                      unassignedPoints: npt.NDArray[np.double],
                      minDistanceThreshold: float,
                      maxDistanceThreshold: float,
                      minAngleThreshold: float,
                      maxAngleThreshold: float,
                      resolution: float,
                      windowSize: float,
                      densityFactor: float
                      ) -> Tuple[Branch, List[Branch], npt.NDArray[np.double]]:
    rootComponent = ConnectedComponent(list(breadthFirstSearch(rootComponentStem)))
    unconnectedComponents = [UnconnectedComponent(set(breadthFirstSearch(stem)))
                             for stem in unconnectedComponentStems if stem is not rootComponentStem]

    originalComponentCount = len(unconnectedComponents) + 1

    allBranches = list(chain(rootComponent.branches,
                             chain.from_iterable([c.branches for c in unconnectedComponents])))

    assignedPoints = np.concatenate([b.tlsPoints for b in allBranches], axis=0)
    assignedPointsLookUp = KDTree(assignedPoints)

    unassignedPointsLookUp = KDTree(unassignedPoints)

    originalMaxDistanceThreshold = maxDistanceThreshold
    maxCheckedDistance = 0.0
    currentComponent: ConnectedComponent = rootComponent

    while unconnectedComponents:
        currentPoints, searchRanges = currentComponent.getPoints(assignedPointsLookUp)
        currentPointsLookUp = KDTree(currentPoints)
        for otherComponent in unconnectedComponents:
            if currentComponent.getBoundingBoxDistance(otherComponent) <= maxDistanceThreshold:
                findConnections(currentComponent, otherComponent, allBranches, searchRanges,
                                currentPointsLookUp, unassignedPointsLookUp,
                                minDistanceThreshold, maxDistanceThreshold, maxCheckedDistance,
                                minAngleThreshold, maxAngleThreshold,
                                windowSize, densityFactor)

        if currentComponent is not rootComponent:
            rootComponent.extend(currentComponent)

        connectableComponents = [c for c in unconnectedComponents if len(c.possibleConnections) > 0]
        if len(connectableComponents) > 0:
            connectionComponent = max(connectableComponents, key=lambda c: c.possibleConnections[0])

            del(unconnectedComponents[unconnectedComponents.index(connectionComponent)])
            currentComponent, connectionBranch = connectionComponent.connect(resolution)

            for component in unconnectedComponents:
                if component.getBoundingBoxDistance(currentComponent) == 0:
                    component.updatePossibleConnections(connectionBranch, minAngleThreshold, minDistanceThreshold)

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

    trimmedPoints1 = Branch.reassignPoints(list(rootComponent.branches), useTrimming=True)
    allBranches = list(chain(rootComponent.branches,
                             chain.from_iterable([c.branches for c in unconnectedComponents])))
    trimmedPoints2 = Branch.assignAllPoints(unassignedPoints, allBranches, useTrimming=True)
    trimmedPoints = np.concatenate((trimmedPoints1, trimmedPoints2), axis=0)

    prunedBranches = list(chain.from_iterable([uc.branches for uc in unconnectedComponents]))

    for branch in rootComponent.branches:
        if branch.isStem():
            return branch, prunedBranches, trimmedPoints
    raise ValueError("No stem was found")


def findConnections(rootComponent: ConnectedComponent,
                    unconnectedComponent: UnconnectedComponent,
                    allBranches: List[Branch],
                    searchRanges: List[npt.NDArray[np.double]],
                    rootPointsLookUp: KDTree,
                    unassignedPointsLookUp: KDTree,
                    minDistanceThreshold: float,
                    maxDistanceThreshold: float,
                    maxCheckedDistance: float,
                    minAngleThreshold: float,
                    maxAngleThreshold: float,
                    windowSize: float,
                    densityFactor: float):
    for endBranch in unconnectedComponent.branches:
        if rootComponent.getBoundingBoxDistance(endBranch) > maxDistanceThreshold:
            continue
        directionsAtEnd = unconnectedComponent.getDirectionData(endBranch, minDistanceThreshold)

        for startBranch, searchRange in zip(rootComponent.branches, searchRanges):
            if startBranch.branchBoundingBoxDistance(endBranch) > maxDistanceThreshold:
                continue
            directionsAtStart = rootComponent.getDirectionData(startBranch, minDistanceThreshold)

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
                    if score is not None:
                        connection = (score, startBranch, startIdx, endBranch, endIdx)
                        unconnectedComponent.possibleConnections.append(connection)
    unconnectedComponent.possibleConnections.sort(key=lambda c: c[0], reverse=True)


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
                    ) -> Optional[float]:
    startPoint = startBranch.skeletonPoints[startIdx]
    endPoint = endBranch.skeletonPoints[endIdx]
    connectionVector = endPoint - startPoint
    connectionDirection = normalizeVectors(connectionVector)
    connectionLength = np.linalg.norm(connectionVector)

    if not areConnectionAnglesCredible(connectionDirection,
                                       parentDirections, siblingDirections, startDirection, childDirections,
                                       minAngleThreshold, maxAngleThreshold):
        return None

    midPoint = startPoint + 0.5*connectionVector
    searchRadius = 0.5*connectionLength + 2*searchRange
    closeRootPoints = rootPointsLookUp.query_ball_point(midPoint, searchRadius)
    closeUnassignedPoints = unassignedPointsLookUp.query_ball_point(midPoint, searchRadius)
    closePoints = np.concatenate((closeRootPoints, closeUnassignedPoints), axis=0)

    _, branchDistances = Branch.findLocalClosestBranches(closePoints, allBranches,
                                                         [startBranch, endBranch])

    pedalDistances, pedalPoints, _ = stripPointsDistances(np.stack((startPoint, endPoint)), closePoints,
                                                          returnPedalInfo=True)

    pointIsAtConnection = pedalDistances < branchDistances & pedalDistances < 2 * searchRange
    connectionPoints = closePoints[pointIsAtConnection]
    pedalPoints = pedalPoints[pointIsAtConnection]
    pedalDistances = pedalDistances[pointIsAtConnection]

    if (len(connectionPoints) == 0 or
            not isConnectionDensityCredible(startPoint, endPoint, pedalPoints, windowSize, densityFactor)):
        return None

    score = len(connectionPoints) / connectionLength / np.mean(pedalDistances)

    return score


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
