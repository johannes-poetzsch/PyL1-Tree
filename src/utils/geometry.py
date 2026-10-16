import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from scipy.stats import ttest_ind

from typing import List, Literal, Optional, Tuple
from src.utils.gpuAcceleration import array_module

USE_FAST_POINT_DISTANCES = False


def setFastStripPointDistances(enabled: bool) -> None:
    global USE_FAST_POINT_DISTANCES
    USE_FAST_POINT_DISTANCES = enabled


def angle(directions_a: npt.NDArray,
          directions_b: npt.NDArray,
          vectorsAreNormalized: bool = True) -> npt.NDArray:
    xp = array_module(directions_a, directions_b)
    directions_a, directions_b = xp.asarray(directions_a), xp.asarray(directions_b)
    if not vectorsAreNormalized:
        directions_a = normalizeVectors(directions_a)
        directions_b = normalizeVectors(directions_b)
    cos = xp.einsum('...d,...d->...', directions_a, directions_b)
    cos = xp.clip(cos, -1.0, 1.0)
    return xp.arccos(cos)


def distanceAlongDirection(startPoints: npt.NDArray[np.double],
                           endPoints: npt.NDArray[np.double],
                           direction: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    xp = array_module(startPoints, endPoints, direction)
    startPoints, endPoints, direction = xp.asarray(startPoints), xp.asarray(endPoints), xp.asarray(direction)
    return xp.einsum('...d,...d->...', endPoints - startPoints, direction)


def findClosestCylinders(axisStarts: npt.NDArray[np.double],
                         axisEnds: npt.NDArray[np.double],
                         radii: List[float] | npt.NDArray[float],
                         points: npt.NDArray[np.double]
                         ) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.double]]:
    startLookUp = KDTree(axisStarts)
    endLookUp = KDTree(axisEnds)

    _, closestStarts = startLookUp.query(points)
    _, closestEnds = endLookUp.query(points)

    startCylinderDistances = cylinderPointsDistance(axisStarts[closestStarts], axisEnds[closestStarts],
                                                    radii[closestStarts], points)
    endCylinderDistances = cylinderPointsDistance(axisStarts[closestEnds], axisEnds[closestEnds],
                                                  radii[closestEnds], points)

    closerToStart = startCylinderDistances < endCylinderDistances
    cylinderDistances = np.where(closerToStart, startCylinderDistances, endCylinderDistances)
    closestCylinders = np.where(closerToStart, closestStarts, closestEnds)

    pointsLookUp = KDTree(points)
    maxDistance = cylinderDistances.max()
    cylinderLengths = np.linalg.norm(axisEnds - axisStarts, axis=1)
    for i in range(len(axisStarts)):
        maxAxisDistance = radii[i] + maxDistance
        searchRange = np.sqrt((cylinderLengths[i]/2)**2 + maxAxisDistance**2)
        cylinderPoints = np.stack([axisStarts[i], axisEnds[i]], axis=0)
        localPointsIndices = pointsLookUp.query_ball_point(cylinderPoints, searchRange)

        maxLocalAxisDistance = radii[i] + cylinderDistances[localPointsIndices].max()
        searchRange = np.sqrt((cylinderLengths[i]/2)**2 + maxLocalAxisDistance**2)
        localPointsIndices = np.unique(pointsLookUp.query_ball_point(cylinderPoints, searchRange))

        newDistances = cylinderPointsDistance(axisStarts[localPointsIndices], axisEnds[localPointsIndices],
                                              radii[localPointsIndices], points)

        improved = newDistances < cylinderDistances[localPointsIndices]
        improvedIndices = localPointsIndices[improved]
        cylinderDistances[improvedIndices] = newDistances[improved]
        closestCylinders[improvedIndices] = i

    return closestCylinders, cylinderDistances


def getStripAngles(strip: npt.NDArray[np.double]) -> Optional[npt.NDArray[np.double]]:
    directions = normalizeVectors(strip[1:] - strip[:-1])
    angles = angle(directions[:-1], directions[1:])
    return angles


def getValueOccurrences(array: npt.NDArray[np.intp|int]) -> List[Tuple[int, npt.NDArray[np.intp]]]:
    xp = array_module(array)
    occurrences = []
    order = xp.argsort(array)
    values, indices, counts = xp.unique(array[order], return_index=True, return_counts=True)
    for v, i, c in zip(values, indices, counts):
        occurrences.append((v, order[i:i+c]))

    return occurrences


def cylinderPointsDistance(axisStarts: npt.NDArray[np.double],
                           axisEnds: npt.NDArray[np.double],
                           radius: float,
                           points: npt.NDArray[np.double]) -> float:
    xp = array_module(axisStarts, axisEnds, points)
    axisStarts, axisEnds, points = xp.asarray(axisStarts), xp.asarray(axisEnds), xp.asarray(points)
    axisDistances = linesPointsDistances(axisStarts, axisEnds, points)
    return xp.abs(axisDistances - radius)


def movingTTest(sequence: npt.NDArray[np.double]) -> Optional[int]:
    windowSize = max(5, int(np.floor(len(sequence) / 10)))
    if len(sequence) < 2 * windowSize:
        return None

    for startIdx, doubleWindow in enumerate(np.lib.stride_tricks.sliding_window_view(sequence, 2*windowSize)):
        firstWindow = doubleWindow[:windowSize]
        firstWindow = firstWindow[~np.isnan(firstWindow)]
        secondWindow = doubleWindow[windowSize:]
        secondWindow = secondWindow[~np.isnan(secondWindow)]
        if len(firstWindow) < 2 or len(secondWindow) < 2:
            continue
        _, pValue = ttest_ind(firstWindow, secondWindow, alternative='less')

        if pValue < 0.01: # TODO: try higher Values
            breakIdx = startIdx + windowSize - 1 # break idx is last index of first window
            return breakIdx

    return None


def normalizeVectors(vectors: npt.NDArray):
    xp = array_module(vectors)
    return vectors / xp.expand_dims(xp.linalg.norm(vectors, axis=-1), axis=-1)


def queryPointsNearSegment(lookUp: KDTree,
                           startPoint: npt.NDArray[np.double],
                           endPoint: npt.NDArray[np.double],
                           range: float,
                           spacing: Optional[float] = None
                           ) -> npt.NDArray[np.intp]:
    segmentVector = endPoint - startPoint
    segmentLength = np.linalg.norm(segmentVector)

    # default spacing: 2r (tune down for fewer false positives)
    if spacing is None:
        spacing = 2 * range
    spacing = min(spacing, max(segmentLength, 1e-9))  # don't exceed segment length

    numIntervals = max(1, int(np.ceil(segmentLength / spacing)))
    t = np.linspace(0.0, 1.0, numIntervals + 1)
    samples = startPoint + t[:, None] * (segmentVector)

    actualSpacing = segmentLength / numIntervals
    queryRadius = np.sqrt(range ** 2 + (actualSpacing / 2) ** 2)

    indices = np.concatenate(lookUp.query_ball_point(samples, queryRadius), axis=0)
    return indices.astype(np.intp, copy=False)


def queryPointsNearStrip(lookUp: KDTree,
                         strip: npt.NDArray[np.double],
                         range: float,
                         spacing: Optional[float] = None
                         ) -> npt.NDArray[np.intp]:
    """Return points near any segment of a strip using one batched KD-tree query."""
    if range <= 0 or len(strip) < 2:
        return np.empty(0, dtype=np.intp)

    segmentVectors = strip[1:] - strip[:-1]
    segmentLengths = np.linalg.norm(segmentVectors, axis=1)

    samplePoints = []
    queryRadii = []
    for segmentIdx, (startPoint, endPoint, segmentVector, segmentLength) in enumerate(
            zip(strip[:-1], strip[1:], segmentVectors, segmentLengths)):
        segmentSpacing = 2 * range if spacing is None else spacing
        segmentSpacing = min(segmentSpacing, max(segmentLength, 1e-9))

        numIntervals = max(1, int(np.ceil(segmentLength / segmentSpacing)))
        t = np.linspace(0.0, 1.0, numIntervals + 1)
        samplesForSegment = startPoint + t[:, None] * segmentVector

        actualSpacing = segmentLength / numIntervals
        radiiForSegment = np.full(numIntervals + 1, np.sqrt(range ** 2 + (actualSpacing / 2) ** 2), dtype=np.double)

        if segmentIdx > 0:
            # The shared vertex covers both segment queries, so retain the larger radius that either segment would have used.
            queryRadii[-1][-1] = max(queryRadii[-1][-1], radiiForSegment[0])
            samplesForSegment = samplesForSegment[1:]
            radiiForSegment = radiiForSegment[1:]

        samplePoints.append(samplesForSegment)
        queryRadii.append(radiiForSegment)

    samples = np.concatenate(samplePoints, axis=0)
    radii = np.concatenate(queryRadii)
    workers = -1 if len(samples) >= 32 else 1
    neighbors = lookUp.query_ball_point(samples, radii, workers=workers, return_sorted=False)
    return np.asarray(np.concatenate(neighbors, axis=0), dtype=np.intp)


def pointSort(points: npt.NDArray,
              order: Tuple[Literal['x', 'y', 'z'], Literal['x', 'y', 'z'], Literal['x', 'y', 'z']] = ('x', 'y', 'z')
              ) -> npt.NDArray:
    xp = array_module(points)
    return xp.sort(points.view([('x', points.dtype), ('y', points.dtype), ('z', points.dtype)]).squeeze(-1),
                   order=order).view((points.dtype, 3))


def linesPointsDistances(lineStartPoints: npt.NDArray[np.double], # (L,D)
                         lineEndPoints: npt.NDArray[np.double], # (L,D) or (D)
                         points: npt.NDArray[np.double] # (N,D)
                         ) -> npt.NDArray[np.double]:
    xp = array_module(lineStartPoints, lineEndPoints, points)
    lineStartPoints, lineEndPoints, points = xp.asarray(lineStartPoints), xp.asarray(lineEndPoints), xp.asarray(points)
    if len(lineStartPoints) == 0:
        return xp.empty((0, len(points)), dtype=xp.double)
    if USE_FAST_POINT_DISTANCES:
        return _linesPointsDistancesFast(lineStartPoints, lineEndPoints, points)

    return _linesPointsDistancesReference(lineStartPoints, lineEndPoints, points)


def _linesPointsDistancesReference(lineStartPoints: npt.NDArray,
                                   lineEndPoints: npt.NDArray,
                                   points: npt.NDArray
                                   ) -> npt.NDArray:
    xp = array_module(lineStartPoints, lineEndPoints, points)
    lineVectors = lineStartPoints - lineEndPoints # (L,D)
    lineLengthsSquared = xp.einsum('LD,LD->L', lineVectors, lineVectors)

    pointsToLineStartPoints = lineStartPoints[:, xp.newaxis] - points # (L,N,D)
    projection = (xp.einsum('LND,LD->LN', pointsToLineStartPoints, lineVectors)
                  / lineLengthsSquared[:, xp.newaxis]) # (L,N)

    xp.clip(projection, 0, 1, out=projection)
    pedalPoints = (lineStartPoints[:, xp.newaxis]
                   - projection[:, :, xp.newaxis] * lineVectors[:, xp.newaxis]) # (L,N,D)

    distances = xp.linalg.norm(pedalPoints - points, axis=-1) # (L,N)

    return distances


def _linesPointsDistancesFast(lineStartPoints: npt.NDArray[np.double],
                              lineEndPoints: npt.NDArray[np.double],
                              points: npt.NDArray[np.double]
                              ) -> npt.NDArray[np.double]:
    xp = array_module(lineStartPoints, lineEndPoints, points)
    segmentVectors = lineEndPoints - lineStartPoints
    segmentLengthsSquared = xp.sum(segmentVectors * segmentVectors, axis=1)

    origin = lineStartPoints[0]
    localStarts = lineStartPoints - origin
    localStartLengthsSquared = xp.sum(localStarts * localStarts, axis=1)
    localStartSegmentDots = xp.sum(localStarts * segmentVectors, axis=1)

    localPoints = points - origin
    pointLengthsSquared = xp.sum(localPoints * localPoints, axis=1)

    pointStartDots = localPoints @ localStarts.T
    pointSegmentDots = localPoints @ segmentVectors.T - localStartSegmentDots
    startDistancesSquared = (pointLengthsSquared[xp.newaxis, :]
                             + localStartLengthsSquared[:, xp.newaxis]
                             - 2 * pointStartDots.T)

    projection = pointSegmentDots.T / segmentLengthsSquared[:, xp.newaxis]
    clippedProjection = xp.clip(projection, 0.0, 1.0)
    segmentDistancesSquared = (startDistancesSquared
                               - 2 * clippedProjection * pointSegmentDots.T
                               + clippedProjection * clippedProjection * segmentLengthsSquared[:, xp.newaxis])
    xp.maximum(segmentDistancesSquared, 0.0, out=segmentDistancesSquared)
    distances = xp.sqrt(segmentDistancesSquared)

    return distances


def stripPointsDistances(strip: npt.NDArray,  # (L+1,D)
                         points: npt.NDArray,  # (N',D)
                         trimStart = False,
                         trimEnd = False,
                         returnTrimming = False,
                         returnPedalInfo = False,
                         maxBatchSize = 1024
                         ) -> (Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.intp], npt.NDArray[np.bool]] |
                               Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.intp]] |
                               Tuple[npt.NDArray[np.double], npt.NDArray[np.bool]] |
                               npt.NDArray[np.double]):
    if len(strip) < 2:
        raise ValueError('Strip must have at least 2 points')

    xp = array_module(strip, points)
    strip, points = xp.asarray(strip), xp.asarray(points)
    segmentVectors = strip[1:] - strip[:-1]
    if xp.any(xp.all(segmentVectors == 0, axis=1)):
        raise ValueError('Strip must not contain zero-length segments')

    if USE_FAST_POINT_DISTANCES and not returnPedalInfo:
        return _stripPointDistancesOnly(strip, points, trimStart, trimEnd, returnTrimming, maxBatchSize)

    return _stripPointsDistancesWithInfo(strip, points, trimStart, trimEnd, returnTrimming,
                                         returnPedalInfo, maxBatchSize)


def _stripPointsDistancesWithInfo(strip: npt.NDArray,
                                  points: npt.NDArray,
                                  trimStart: bool,
                                  trimEnd: bool,
                                  returnTrimming: bool,
                                  returnPedalInfo: bool,
                                  maxBatchSize: int
                                  ) -> (Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.intp], npt.NDArray[np.bool]] |
                                        Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.intp]] |
                                        Tuple[npt.NDArray[np.double], npt.NDArray[np.bool]] |
                                        npt.NDArray[np.double]):
    if len(strip) < 2:
        raise ValueError('Strip must have at least 2 points')
    if maxBatchSize <= 0:
        raise ValueError("maxBatchSize must be greater than zero")
    xp = array_module(strip, points)
    strip, points = xp.asarray(strip), xp.asarray(points)
    segmentStarts = strip[:-1]
    segmentEnds = strip[1:]
    segmentVectors = segmentStarts - segmentEnds # (L,D)

    distances = xp.empty(len(points), dtype=np.double)
    pedalPoints = xp.empty_like(points, dtype=np.double)
    closestSegments = xp.empty(len(points), dtype=np.intp)
    trimmed = xp.zeros(len(points), dtype=np.bool)

    for start in range(0, len(points), maxBatchSize):
        b = slice(start, min(start + maxBatchSize, len(points)))

        pointsToSegmentStarts = segmentStarts - points[b,xp.newaxis] # (N,L,D)
        projection = (xp.einsum('NLD,LD->NL', pointsToSegmentStarts, segmentVectors)
                      / xp.einsum('LD,LD->L', segmentVectors, segmentVectors)) # (N,L)
        clippedProjection = xp.clip(projection, 0.0, 1.0)
        potentialPedalPoints = segmentStarts[xp.newaxis] - clippedProjection[:,:,xp.newaxis] * segmentVectors[xp.newaxis] # (N,L,D)

        segmentDistances: npt.NDArray = xp.linalg.norm(potentialPedalPoints - points[b,xp.newaxis], axis=-1) # (N,L)
        closestSegments[b] = segmentDistances.argmin(axis=-1) # (N)
        distances[b] = xp.take_along_axis(segmentDistances, closestSegments[b,xp.newaxis], axis=-1).squeeze() # (N)

        if returnPedalInfo:
            pedalPoints[b] = xp.take_along_axis(potentialPedalPoints, closestSegments[b,xp.newaxis,xp.newaxis], axis=1).squeeze() # (N,D)
        if trimStart:
            trimmed[b] |= (closestSegments[b] == 0) & (projection[:,0] < 0)
        if trimEnd:
            endIdx = len(segmentVectors) - 1
            trimmed[b] |= (closestSegments[b] == endIdx) & (projection[:,endIdx] > 1)

    if returnPedalInfo:
        if returnTrimming:
            return distances, pedalPoints, closestSegments, trimmed
        else:
            untrimmed = ~trimmed
            return distances[untrimmed], pedalPoints[untrimmed], closestSegments[untrimmed]
    else:
        if returnTrimming:
            return distances, trimmed
        else:
            return distances[~trimmed]


def _stripPointDistancesOnly(strip: npt.NDArray,
                             points: npt.NDArray,
                             trimStart: bool,
                             trimEnd: bool,
                             returnTrimming: bool,
                             maxBatchSize: int
                             ) -> (Tuple[npt.NDArray[np.double], npt.NDArray[np.bool]] | npt.NDArray[np.double]):
    if len(strip) < 2:
        raise ValueError('Strip must have at least 2 points')
    if maxBatchSize <= 0:
        raise ValueError("maxBatchSize must be greater than zero")

    xp = array_module(strip, points)
    strip, points = xp.asarray(strip), xp.asarray(points)
    segmentStarts = strip[:-1]
    segmentVectors = strip[1:] - segmentStarts
    segmentLengthsSquared = xp.sum(segmentVectors * segmentVectors, axis=1)

    distances = xp.empty(len(points), dtype=xp.double)
    trimmed = xp.zeros(len(points), dtype=xp.bool_)
    origin = segmentStarts[0]
    localStarts = segmentStarts - origin
    localStartLengthsSquared = xp.sum(localStarts * localStarts, axis=1)
    localStartSegmentDots = xp.sum(localStarts * segmentVectors, axis=1)

    for start in range(0, len(points), maxBatchSize):
        b = slice(start, min(start + maxBatchSize, len(points)))
        localPoints = points[b] - origin
        pointLengthsSquared = xp.sum(localPoints * localPoints, axis=1)

        pointStartDots = localPoints @ localStarts.T
        pointSegmentDots = localPoints @ segmentVectors.T - localStartSegmentDots
        startDistancesSquared = (pointLengthsSquared[:,xp.newaxis]
                                 + localStartLengthsSquared[xp.newaxis,:]
                                 - 2 * pointStartDots)

        projection = pointSegmentDots / segmentLengthsSquared[xp.newaxis,:]
        clippedProjection = xp.clip(projection, 0.0, 1.0)
        segmentDistancesSquared = (startDistancesSquared
                                   - 2 * clippedProjection * pointSegmentDots
                                   + clippedProjection * clippedProjection * segmentLengthsSquared[xp.newaxis,:])
        xp.maximum(segmentDistancesSquared, 0.0, out=segmentDistancesSquared)

        closestSegments = xp.argmin(segmentDistancesSquared, axis=1)
        closestDistancesSquared = xp.take_along_axis(segmentDistancesSquared, closestSegments[:,xp.newaxis], axis=1)[:,0]
        distances[b] = xp.sqrt(closestDistancesSquared)

        if trimStart:
            trimmed[b] |= (closestSegments == 0) & (projection[:,0] < 0)
        if trimEnd:
            endIdx = len(segmentVectors) - 1
            trimmed[b] |= ((closestSegments == endIdx) & (projection[:,endIdx] > 1))

    if returnTrimming:
        return distances, trimmed
    return distances[~trimmed]
