from __future__ import annotations
from itertools import product
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from src.objects.boundedObject import BoundedObject
from src.objects.branch import Branch
from src.objects.connectedComponent import ConnectedComponent
from src.objects.tlsPointManager import TlsPointManager
from src.utils.geometry import angle, normalizeVectors, stripPointsDistances


class Connection(BoundedObject):
    def __init__(self,
                 strip: npt.NDArray[np.double],
                 tlsManager: TlsPointManager,
                 startBranch: Branch,
                 startIdx: int,
                 startComponent: ConnectedComponent,
                 endBranch: Branch,
                 endIdx: int,
                 startDirection: npt.NDArray[np.double],
                 childDirections: npt.NDArray[np.double],
                 minDistanceThreshold: float,
                 minAngleThreshold: float,
                 maxAngleThreshold: float,
                 windowSize: float,
                 densityFactor: float,
                 stepSize: Optional[float] = None):
        self.isCredible = True

        self.strip = strip
        self.tlsManager = tlsManager
        self.startBranch = startBranch
        self.startIdx = startIdx
        self.startComponent = startComponent
        self.endBranch = endBranch
        self.endIdx = endIdx

        self.startDirection = startDirection
        self.childDirections = childDirections

        self.minDistanceThreshold = minDistanceThreshold
        self.minAngleThreshold = minAngleThreshold
        self.maxAngleThreshold = maxAngleThreshold
        self.windowSize = windowSize
        self.densityFactor = densityFactor
        self.stepSize = stepSize

        connectionDirection = normalizeVectors(strip[-1] - strip[0])
        self.connectionDirectionsAtStart, self.connectionDirectionsAtEnd = [connectionDirection], [connectionDirection]

        if len(strip) > 2:
            self.connectionDirectionsAtStart.append(normalizeVectors(strip[1] - strip[0]))
            self.connectionDirectionsAtEnd.append(normalizeVectors(strip[-1] - strip[-2]))

        if not self.anglesAreCredible():
            self.isCredible = False
            return

        assignedIds, unassignedIds = tlsManager.getPointsAtStrip(strip, startBranch)
        self.assignedTlsIds, self.aNewDistances, self.unassignedTlsIds, self.uaNewDistances, pedalPoints, segmentIds = (
            self.findImprovedPoints(assignedIds, unassignedIds, 2 * self.tlsManager.searchRanges[startBranch]))

        if not self.densityIsCredible(pedalPoints, segmentIds):
            self.isCredible = False
            return

        boundingBoxMin, boundingBoxMax = self.getBoundingBox()
        BoundedObject.__init__(self, boundingBoxMin, boundingBoxMax)


        self.score = self.calculateScore()


    def apply(self, resolution: float):
        parent, sibling = self.tlsManager.splitBranch(self.startBranch, self.startIdx)
        if parent is None:
            raise ValueError("Can not connect to trunk at idx 0.")
        if sibling is not None:
            self.startComponent.branches.append(sibling)
            del self.startComponent.directionData[parent]

        # the original tlsIds only included points in a reasonable radius around the connection
        # however, there might be other improved points outside of this range
        self.assignedTlsIds, self.aNewDistances, self.unassignedTlsIds, self.uaNewDistances, _, _ = self.findImprovedPoints(
            TlsPointManager.findPointIndicesAroundStrip(
                self.strip, self.tlsManager.assignedPointsLookUp,
                self.tlsManager.assignedPointsDistances, self.tlsManager.assignedPointsMaxDistance),
            TlsPointManager.findPointIndicesAroundStrip(
                self.strip, self.tlsManager.unassignedPointsLookUp,
                self.tlsManager.unassignedPointsDistances, self.tlsManager.unassignedPointsMaxDistance))

        self.boundingBoxMin, self.boundingBoxMax = self.getBoundingBox()

        connectionBranch = self.tlsManager.createBranch(self.strip, parent, self.assignedTlsIds, self.aNewDistances,
                                                        self.unassignedTlsIds, self.uaNewDistances, resolution)

        return connectionBranch, sibling


    def anglesAreCredible(self) -> bool:
        if not self.isCredible:
            return False

        directionsAtStartBranch = self.startComponent.getDirectionData(self.startBranch, self.minDistanceThreshold)
        parentDirections, siblingDirections = directionsAtStartBranch[self.startIdx]

        for connectionDirection in self.connectionDirectionsAtEnd:
            if angle(self.startDirection, -connectionDirection) > self.maxAngleThreshold:
                return False

        for parentDirections, connectionDirection in product(parentDirections, self.connectionDirectionsAtStart):
            if angle(parentDirections, -connectionDirection) > self.maxAngleThreshold:
                return False

        for siblingDirection, connectionDirection in product(siblingDirections, self.connectionDirectionsAtStart):
            if angle(siblingDirection, connectionDirection) <= self.minAngleThreshold:
                return False

        for childDirection, connectionDirection in product(self.childDirections, self.connectionDirectionsAtEnd):
            if (angle(childDirection, connectionDirection) > self.maxAngleThreshold
                    or angle(childDirection, -connectionDirection) <= self.minAngleThreshold):
                return False

        return True


    def calculateScore(self) -> float:
        connectionLength = np.sum(np.linalg.norm(self.strip[1:] - self.strip[:-1], axis=1))
        distances = np.append(self.aNewDistances, self.uaNewDistances, axis=0)

        return len(distances) / connectionLength / distances.mean()


    def densityIsCredible(self,
                          pedalPoints: npt.NDArray[np.double],
                          segmentIds: npt.NDArray[np.intp]) -> bool:
        if not self.isCredible or len(pedalPoints) == 0:
            return False

        distancesOnSegment = np.linalg.norm(pedalPoints - self.strip[segmentIds], axis=1)
        segmentLengths = np.linalg.norm(self.strip[1:] - self.strip[:-1], axis=1)
        lengthsUntilSegmentEnds = np.cumsum(segmentLengths)
        totalLength = lengthsUntilSegmentEnds[-1]
        if totalLength < self.windowSize:
            return True

        lengthsBeforeSegments = lengthsUntilSegmentEnds - segmentLengths
        distancesToStart = lengthsBeforeSegments[segmentIds] + distancesOnSegment

        minPointsPerWindow = int(np.ceil(self.windowSize / totalLength * self.densityFactor * len(pedalPoints)))
        if self.stepSize is None:
            if len(pedalPoints) < minPointsPerWindow:
                return False

            distancesToStart = np.sort(np.append(distancesToStart, (0, totalLength)))

            # Excluding the start but including the end, each interval contains exactly minPointsPerWindow points.
            # If the distance between interval start and interval end is greater than windowSize,
            # that means there is a window that starts right after the interval start and ends before its end.
            # This window thus contains less than minPointsPerWindow points and violates the density constrain
            intervalLengths = distancesToStart[minPointsPerWindow:] - distancesToStart[:-minPointsPerWindow]
            if np.any(intervalLengths > self.windowSize):
                return False
        else:
            distancesToStart = np.sort(distancesToStart)

            windowEnds = np.arange(self.windowSize, totalLength, self.stepSize, dtype=np.double)
            windowStarts = windowEnds - self.windowSize
            pointsPerWindow = np.searchsorted(distancesToStart, windowEnds) - np.searchsorted(distancesToStart, windowStarts)

            if np.any(pointsPerWindow < minPointsPerWindow):
                return False

        return True


    def findImprovedPoints(self,
                           assignedIds: npt.NDArray[np.intp],
                           unassignedIds: npt.NDArray[np.intp],
                           maxDistance: Optional[float] = None):
        assignedIds, aNewDistances, aPedalPoints, aSegmentIds = Connection.getImprovedData(
            self.strip, assignedIds, self.tlsManager.assignedPoints[assignedIds],
            self.tlsManager.assignedPointsDistances[assignedIds], maxDistance)

        unassignedIds, uaNewDistances, uaPedalPoints, uaSegmentIds = Connection.getImprovedData(
            self.strip, unassignedIds, self.tlsManager.unassignedPoints[unassignedIds],
            self.tlsManager.unassignedPointsDistances[unassignedIds], maxDistance)

        pedalPoints = np.append(aPedalPoints, uaPedalPoints, axis=0)
        segmentIds = np.append(aSegmentIds, uaSegmentIds)

        return assignedIds, aNewDistances, unassignedIds, uaNewDistances, pedalPoints, segmentIds


    def getBoundingBox(self):
        assignedPoints = self.tlsManager.assignedPoints[self.assignedTlsIds]
        unassignedPoints = self.tlsManager.unassignedPoints[self.unassignedTlsIds]

        boundingBoxMin = np.min(np.append(assignedPoints, unassignedPoints, axis=0), axis=0)
        boundingBoxMax = np.max(np.append(assignedPoints, unassignedPoints, axis=0), axis=0)

        return boundingBoxMin, boundingBoxMax


    def update(self, realizedConnection: Connection, connectionBranch: Branch):
        if self.startBranch is realizedConnection.startBranch:
            if self.startIdx > realizedConnection.startIdx:
                self.startIdx = self.startIdx - len(self.startBranch) + 1
                self.startBranch = [c for c in self.startBranch.children if c is not connectionBranch][0]

            self.isCredible = self.anglesAreCredible()

        if not self.isCredible:
            return

        if self.getObjectBBDistance(realizedConnection) == 0:
            self.assignedTlsIds, self.aNewDistances, self.unassignedTlsIds, self.uaNewDistances, pedalPoints, segmentIds = (
                self.findImprovedPoints(self.assignedTlsIds, self.unassignedTlsIds))

            self.isCredible = self.densityIsCredible(pedalPoints, segmentIds)

        if self.isCredible:
            self.score = self.calculateScore()


    @staticmethod
    def getImprovedData(strip: npt.NDArray[np.double],
                        ids: npt.NDArray[np.intp],
                        points: npt.NDArray[np.double],
                        distances: npt.NDArray[np.double],
                        maxDistance: Optional[float]
                        ) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.intp]]:
        newDistances, pedalPoints, segmentIds = stripPointsDistances(strip, points, returnPedalInfo=True)

        isAtConnection = newDistances < distances
        if maxDistance is not None:
            isAtConnection &= newDistances < maxDistance

        ids, newDistances = ids[isAtConnection], newDistances[isAtConnection]
        pedalPoints, segmentIds = pedalPoints[isAtConnection], segmentIds[isAtConnection]

        return ids, newDistances, pedalPoints, segmentIds
