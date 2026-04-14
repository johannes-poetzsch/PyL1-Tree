from itertools import product
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Optional, Tuple

from src.utils.geometry import angle, normalizeVectors, stripPointsDistances


def scoreConnection(connectionStrip: npt.NDArray[np.double],
                    searchRange: float,
                    startDirection: npt.NDArray[np.double],
                    childDirections: npt.NDArray[np.double],
                    parentDirections: npt.NDArray[np.double],
                    siblingDirections: npt.NDArray[np.double],
                    rootPoints: npt.NDArray[np.double],
                    rootDistances: npt.NDArray[np.double],
                    unassignedPointsLookUp: KDTree,
                    unassignedPointsDistances: npt.NDArray[np.double],
                    minAngleThreshold: float,
                    maxAngleThreshold: float,
                    windowSize: float,
                    densityFactor: float
                    ) -> Optional[float]:
    connectionDirection = normalizeVectors(connectionStrip[-1] - connectionStrip[0])
    connectionDirectionsAtStart, connectionDirectionsAtEnd = [connectionDirection], [connectionDirection]
    if len(connectionStrip) > 2:
        connectionDirectionsAtStart.append(normalizeVectors(connectionStrip[1] - connectionStrip[0]))
        connectionDirectionsAtEnd.append(normalizeVectors(connectionStrip[-1] - connectionStrip[-2]))

    if not areConnectionAnglesCredible(np.array(connectionDirectionsAtStart), np.array(connectionDirectionsAtEnd),
                                       parentDirections, siblingDirections, startDirection, childDirections,
                                       minAngleThreshold, maxAngleThreshold):
        return None

    rootPoints, rootPedalPoints, rootDistances, rootSegmentIds = findPointsAtConnection(connectionStrip, rootPoints,
                                                                                        rootDistances, searchRange)

    uaPoints, uaDistances = searchPointsAroundStrip(connectionStrip, unassignedPointsLookUp, unassignedPointsDistances,
                                                    searchRange)
    uaPoints, uaPedalPoints, uaDistances, uaSegmentIds = findPointsAtConnection(connectionStrip, uaPoints, uaDistances,
                                                                                2 * searchRange)

    connectionPoints = np.append(rootPoints, uaPoints, axis=0)
    pedalPoints = np.append(rootPedalPoints, uaPedalPoints, axis=0)
    distances = np.append(rootDistances, uaDistances, axis=0)
    segmentIds = np.append(rootSegmentIds, uaSegmentIds, axis=0)

    if (len(connectionPoints) == 0 or
            not isConnectionDensityCredible(connectionStrip, pedalPoints, segmentIds, windowSize, densityFactor)):
        return None

    connectionLength = np.sum(np.linalg.norm(connectionStrip[1:] - connectionStrip[:-1], axis=1))
    score = len(connectionPoints) / connectionLength / distances.mean()

    return score


def areConnectionAnglesCredible(connectionDirectionsAtStart: npt.NDArray[np.double],
                                connectionDirectionsAtEnd: npt.NDArray[np.double],
                                parentDirections: Optional[npt.NDArray[np.double]] = None,
                                siblingDirections: Optional[npt.NDArray[np.double]] = None,
                                startDirection: Optional[npt.NDArray[np.double]] = None,
                                childDirections: Optional[npt.NDArray[np.double]] = None,
                                minAngleThreshold: float = 0.0,
                                maxAngleThreshold: float = 0.0
                                ) -> bool:
    if len(connectionDirectionsAtEnd) > 0:
        startToConnectionAngles = angle(startDirection, -connectionDirectionsAtEnd)
        if np.any(startToConnectionAngles > maxAngleThreshold):
            return False

    if parentDirections is not None:
        for parentDirections, connectionDirection in product(parentDirections, connectionDirectionsAtStart):
            if angle(parentDirections, -connectionDirection) > maxAngleThreshold:
                return False

    if siblingDirections is not None:
        for siblingDirection, connectionDirection in product(siblingDirections, connectionDirectionsAtStart):
            if angle(siblingDirection, connectionDirection) <= minAngleThreshold:
                return False

    if childDirections is not None:
        for childDirection, connectionDirection in product(childDirections, connectionDirectionsAtEnd):
            if (angle(childDirection, connectionDirection) > maxAngleThreshold
                    or angle(childDirection, -connectionDirection) <= minAngleThreshold):
                return False

    return True


def findPointsAtConnection(connectionStrip: npt.NDArray[np.double],
                           points: npt.NDArray[np.double],
                           pointDistances: npt.NDArray[np.double],
                           searchRange: float):
    distances, pedalPoints, segmentIds, _ = stripPointsDistances(connectionStrip, points, returnPedalInfo=True)
    isAtConnection = (distances < pointDistances) & (distances < 2 * searchRange)

    return points[isAtConnection], pedalPoints[isAtConnection], distances[isAtConnection], segmentIds[isAtConnection]


def isConnectionDensityCredible(connectionStrip: npt.NDArray[np.double],
                                pedalPoints: npt.NDArray[np.double],
                                segmentIds: npt.NDArray[np.intp],
                                windowSize: float,
                                densityFactor: float) -> bool:
    distancesOnSegment = np.linalg.norm(pedalPoints - connectionStrip[segmentIds], axis=1)
    segmentLengths = np.linalg.norm(connectionStrip[1:] - connectionStrip[:-1], axis=1)
    lengthsUntilSegmentEnds = np.cumsum(segmentLengths)
    totalLength = lengthsUntilSegmentEnds[-1]

    if totalLength < windowSize:
        return True
    
    minPointsPerWindow = int(np.ceil(windowSize / totalLength * densityFactor * len(pedalPoints)))
    if len(pedalPoints) < minPointsPerWindow:
        return False

    lengthsBeforeSegments = lengthsUntilSegmentEnds - segmentLengths
    distancesToStart = lengthsBeforeSegments[segmentIds] + distancesOnSegment
    distancesToStart = np.sort(np.append(distancesToStart, (0, totalLength)))


    # Excluding the start but including the end, each interval contains exactly minPointsPerWindow points.
    # If the distance between interval start and interval end is greater than windowSize,
    # that means there is a window that starts right after the interval start and ends before its end.
    # This window thus contains less than minPointsPerWindow points and violates the density constrain
    intervalLengths = distancesToStart[minPointsPerWindow:] - distancesToStart[:-minPointsPerWindow]
    if np.any(intervalLengths > windowSize):
        return False
    else:
        return True


def searchPointsAroundStrip(connectionStrip: npt.NDArray[np.double],
                            pointsLookUp: KDTree,
                            pointDistances: npt.NDArray[np.double],
                            searchRange: float
                            ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    midPoint = (connectionStrip[0] + connectionStrip[-1]) / 2
    connectionLength = np.linalg.norm(connectionStrip[-1] - connectionStrip[0])
    searchRadius = connectionLength / 2 + searchRange

    closePointIndices = pointsLookUp.query_ball_point(midPoint, searchRadius)
    points = pointsLookUp.data[closePointIndices]
    distances = pointDistances[closePointIndices]

    return points, distances
