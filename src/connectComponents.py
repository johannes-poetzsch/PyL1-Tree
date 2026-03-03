from itertools import chain
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List, Optional, Set, Tuple

from src.objects.branch import Branch
from src.objects.component import ConnectedComponent, UnconnectedComponent
from src.utils.iteration import breadthFirstSearch
from src.utils.geometry import angle, normalizeVectors, stripPointsDistances


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
    unconnectedComponents = {UnconnectedComponent(list(breadthFirstSearch(stem)))
                             for stem in unconnectedComponentStems if stem is not rootComponentStem}

    originalComponentCount = len(unconnectedComponents) + 1

    allBranches = list(chain(breadthFirstSearch(rootComponentStem),
                             chain.from_iterable([c.branches for c in unconnectedComponents])))

    assignedPoints = np.concatenate([b.tlsPoints for b in allBranches], axis=0)
    assignedPointsLookUp = KDTree(assignedPoints)
    unassignedPointsLookUp = KDTree(unassignedPoints)
    unassignedPointsDistances = Branch.findMinBranchDistances(unassignedPoints, allBranches)

    rootComponent = ConnectedComponent(list(breadthFirstSearch(rootComponentStem)), assignedPointsLookUp)
    connectedComponents = {rootComponent}

    originalMaxDistanceThreshold = maxDistanceThreshold
    maxDistanceCap = 2.0
    connectionBranches = set()

    while unconnectedComponents:
        for component in connectedComponents:
            if component.checkedDistanceThreshold < maxDistanceThreshold:
                for otherComponent in unconnectedComponents:
                    if component.getBoundingBoxDistance(otherComponent) <= maxDistanceThreshold:
                        findConnections(component, otherComponent, allBranches, unassignedPointsLookUp,
                                        unassignedPointsDistances, minDistanceThreshold, maxDistanceThreshold,
                                        minAngleThreshold, maxAngleThreshold,
                                        windowSize, densityFactor)

        connectableComponents = [c for c in unconnectedComponents if len(c.possibleConnections) > 0]

        if len(connectableComponents) > 0:
            connectionComponent = max(connectableComponents, key=lambda c: c.possibleConnections[0])

            unconnectedComponents.remove(connectionComponent)
            connectionComponent, connectionBranch, splitEnd = connectionComponent.connect(resolution, assignedPointsLookUp)

            if splitEnd is not None:
                splitBranch = splitEnd.parent
                for component in connectedComponents:
                    if splitBranch in component.directionData:
                        del component.directionData[splitBranch]

            connectedComponents.add(connectionComponent)
            connectionBranches.add((connectionBranch, len(connectionBranch.skeletonPoints)))

            connectionPoint = connectionBranch.skeletonPoints[0]
            for component in unconnectedComponents:
                if component.getBoundingBoxDistance(connectionPoint) <= maxDistanceCap:
                    component.updatePossibleConnections(connectionBranch, minAngleThreshold, minDistanceThreshold)

            maxDistanceThreshold = originalMaxDistanceThreshold
        else:
            if maxDistanceThreshold + 0.2 > maxDistanceCap or len(unconnectedComponents) <= originalComponentCount / 10:
                break
            maxDistanceThreshold += 0.2

    revertInterpolation(connectionBranches)
    for branch in rootComponent.branches:
        if len(branch.children) == 1:
            branch.expand()

    trimmedPoints1 = Branch.reassignPoints(rootComponent.branches, useTrimming=True)
    allBranches = list(chain(rootComponent.branches,
                             chain.from_iterable([c.branches for c in unconnectedComponents])))
    trimmedPoints2 = Branch.assignAllPoints(unassignedPoints, allBranches, useTrimming=True)
    trimmedPoints = np.concatenate((trimmedPoints1, trimmedPoints2), axis=0)

    prunedBranches = list(chain.from_iterable([uc.branches for uc in unconnectedComponents]))

    for branch in rootComponent.branches:
        if branch.isStem():
            return branch, prunedBranches, trimmedPoints
    raise ValueError("No stem was found")


def findConnections(connectedComponent: ConnectedComponent,
                    unconnectedComponent: UnconnectedComponent,
                    allBranches: List[Branch],
                    unassignedPointsLookUp: KDTree,
                    unassignedPointsDistances: npt.NDArray[np.double],
                    minDistanceThreshold: float,
                    maxDistanceThreshold: float,
                    minAngleThreshold: float,
                    maxAngleThreshold: float,
                    windowSize: float,
                    densityFactor: float):
    for endBranch in unconnectedComponent.branches:
        if connectedComponent.getBoundingBoxDistance(endBranch) > maxDistanceThreshold:
            continue
        directionsAtEnd = unconnectedComponent.getDirectionData(endBranch, minDistanceThreshold)

        for startBranch, searchRange in zip(connectedComponent.branches, connectedComponent.searchRanges):
            if startBranch.branchBoundingBoxDistance(endBranch) > maxDistanceThreshold:
                continue
            directionsAtStart = connectedComponent.getDirectionData(startBranch, minDistanceThreshold)

            for endIdx, startDirection, childDirections in directionsAtEnd:
                for startIdx, (parentDirections, siblingDirections) in enumerate(directionsAtStart):
                    if len(directionsAtStart) < len(startBranch.skeletonPoints):
                        startIdx += 1

                    connectionLength = np.linalg.norm(
                        startBranch.skeletonPoints[startIdx] - endBranch.skeletonPoints[endIdx])

                    if (connectionLength > maxDistanceThreshold or
                            connectionLength <= connectedComponent.checkedDistanceThreshold):
                        continue

                    score = scoreConnection(startBranch, startIdx, endBranch, endIdx, connectedComponent, searchRange,
                                            startDirection, childDirections, parentDirections, siblingDirections,
                                            allBranches, unassignedPointsLookUp, unassignedPointsDistances,
                                            minAngleThreshold, maxAngleThreshold, windowSize, densityFactor)
                    if score is not None:
                        connection = (score, startBranch, startIdx, endBranch, endIdx)
                        unconnectedComponent.possibleConnections.append(connection)
    unconnectedComponent.possibleConnections.sort(key=lambda c: c[0], reverse=True)


def scoreConnection(startBranch: Branch,
                    startIdx: int,
                    endBranch: Branch,
                    endIdx: int,
                    rootComponent: ConnectedComponent,
                    searchRange: float,
                    startDirection: npt.NDArray[np.double],
                    childDirections: npt.NDArray[np.double],
                    parentDirections: npt.NDArray[np.double],
                    siblingDirections: npt.NDArray[np.double],
                    allBranches: List[Branch],
                    unassignedPointsLookUp: KDTree,
                    unassignedPointsDistances: npt.NDArray[np.double],
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

    closeRootPointIndices = rootComponent.tlsPointsLookUp.query_ball_point(midPoint, searchRadius)
    closeRootPoints = rootComponent.tlsPointsLookUp.data[closeRootPointIndices]
    rootPedalDistances, rootPedalPoints, _ = stripPointsDistances(np.stack((startPoint, endPoint)), closeRootPoints,
                                                                  returnPedalInfo=True)
    rootPointIsAtConnection = rootPedalDistances < rootComponent.tlsDistances[closeRootPointIndices]

    closeUnassignedPointIndices = unassignedPointsLookUp.query_ball_point(midPoint, searchRadius)
    closeUnassignedPoints = unassignedPointsLookUp.data[closeUnassignedPointIndices]
    unassignedPedalDistances, unassignedPedalPoints, _ = stripPointsDistances(np.stack((startPoint, endPoint)),
                                                                              closeUnassignedPoints,
                                                                              returnPedalInfo=True)
    unassignedPointIsAtConnection = unassignedPedalDistances < unassignedPointsDistances[closeUnassignedPointIndices]

    connectionPoints = np.append(closeRootPoints[rootPointIsAtConnection],
                                 closeUnassignedPoints[unassignedPointIsAtConnection], axis=0)
    pedalPoints = np.append(rootPedalPoints[rootPointIsAtConnection],
                            unassignedPedalPoints[unassignedPointIsAtConnection], axis=0)
    pedalDistances = np.append(rootPedalDistances[rootPointIsAtConnection],
                               unassignedPedalDistances[unassignedPointIsAtConnection], axis=0)

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
    pointsPerWindow = int(np.ceil(len(pedalPoints) / length * densityFactor * windowSize))

    relativeWindowSize = windowSize / length
    mainAxis = vector.argmax()

    projectedPedalPoints = pedalPoints[:,mainAxis]
    maxProjectedWindowSize = relativeWindowSize * vector[mainAxis]

    projectedPedalPoints = np.append(projectedPedalPoints, (startPoint[mainAxis], endPoint[mainAxis]), axis=0)
    projectedPedalPoints.sort()

    projectedWindowSizes = projectedPedalPoints[pointsPerWindow:] - projectedPedalPoints[:-pointsPerWindow]

    isCredible = np.all(projectedWindowSizes <= maxProjectedWindowSize)
    return isCredible


def revertInterpolation(connectionBranches: Set[Branch]):
    for branch, remainingPointCount in connectionBranches:
        while remainingPointCount > 1:
            remainingPointCount -= len(branch.skeletonPoints - 1)
            branch.skeletonPoints = branch.skeletonPoints[[0, -1]]

            # if the branch was split by another connection, find the split end of the branch
            if remainingPointCount > 1:
                for c in branch.children:
                    if c not in connectionBranches:
                        branch = c
