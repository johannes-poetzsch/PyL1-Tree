import cupy as cp
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from scipy.stats import ttest_ind

from typing import List, Literal, Optional, Tuple


def angle(directions_a: npt.NDArray,
          directions_b: npt.NDArray,
          vectorsAreNormalized: bool = True) -> npt.NDArray:
    # TODO: compare to torch vecdot
    if not vectorsAreNormalized:
        directions_a = normalizeVectors(directions_a)
        directions_b = normalizeVectors(directions_b)
    cos = np.vecdot(directions_a, directions_b)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def distanceAlongDirection(startPoints: npt.NDArray[np.double],
                           endPoints: npt.NDArray[np.double],
                           direction: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    return np.vecdot(endPoints - startPoints, direction)


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
    occurrences = []
    order = np.argsort(array)
    values, indices, counts = np.unique(array[order], return_index=True, return_counts=True)
    for v, i, c in zip(values, indices, counts):
        occurrences.append((v, order[i:i+c]))

    return occurrences


def cylinderPointsDistance(axisStarts: npt.NDArray[np.double],
                           axisEnds: npt.NDArray[np.double],
                           radius: float,
                           points: npt.NDArray[np.double]) -> float:
    axisDistances = linesPointsDistances(axisStarts, axisEnds, points)
    return np.abs(axisDistances - radius)


def movingTTest(sequence: npt.NDArray[np.double]) -> Optional[int]:
    windowSize = max(5, int(np.floor(len(sequence) / 10)))
    if len(sequence) < 2 * windowSize:
        return None

    for startIdx, doubleWindow in enumerate(np.lib.stride_tricks.sliding_window_view(sequence, 2*windowSize)):
        firstWindow = doubleWindow[:windowSize]
        firstWindow = firstWindow[~np.isnan(firstWindow)]
        secondWindow = doubleWindow[windowSize:]
        secondWindow = secondWindow[~np.isnan(secondWindow)]
        _, pValue = ttest_ind(firstWindow, secondWindow, alternative='less')

        if pValue < 0.01: # TODO: try higher Values
            breakIdx = startIdx + windowSize - 1 # break idx is last index of first window
            return breakIdx

    return None


def normalizeVectors(vectors: npt.NDArray):
    xp = cp.get_array_module(vectors)
    return vectors / xp.expand_dims(xp.linalg.norm(vectors, axis=-1), axis=-1)


def queryPointsNearSegment(lookUp: KDTree,
                           startPoint: npt.NDArray[np.double],
                           endPoint: npt.NDArray[np.double],
                           range: float,
                           spacing: Optional[float] = None
                           ) -> npt.NDArray[np.double]:
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

    return np.concatenate(lookUp.query_ball_point(samples, queryRadius), axis=0)


def pointSort(points: npt.NDArray,
              order: Tuple[Literal['x', 'y', 'z'], Literal['x', 'y', 'z'], Literal['x', 'y', 'z']] = ('x', 'y', 'z')
              ) -> npt.NDArray:
    return np.sort(points.view([('x', points.dtype), ('y', points.dtype), ('z', points.dtype)]).squeeze(-1),
                   order=order).view((points.dtype, 3))


def linesPointsDistances(lineStartPoints: npt.NDArray, # (L,D)
                         lineEndPoints: npt.NDArray, # (L,D) or (D)
                         points: npt.NDArray, # (N,D)
                         ) -> npt.NDArray | Tuple[npt.NDArray, npt.NDArray]:
    xp = cp.get_array_module(lineStartPoints, lineEndPoints, points)
    lineVectors = lineStartPoints - lineEndPoints # (L,D)

    pointsToLineStartPoints = lineStartPoints[:,xp.newaxis] - points # (L,N,D)
    # TODO: compare runtime to torch vecdot
    tMatrix = xp.einsum('LND,LD->LN', pointsToLineStartPoints, lineVectors) / xp.einsum('LD,LD->L', lineVectors, lineVectors)[:,xp.newaxis] # (L,N)
    del pointsToLineStartPoints
    xp.clip(tMatrix, 0, 1, out=tMatrix)
    pedalPoints = lineStartPoints[:,xp.newaxis] - tMatrix[:,:,xp.newaxis] * lineVectors[:,xp.newaxis] # (L,N,D)
    del tMatrix
    distances = xp.linalg.norm(pedalPoints - points, axis=-1) # (L,N)

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
    xp = cp.get_array_module(strip, points)
    segmentStarts = strip[:-1]
    segmentEnds = strip[1:]
    segmentVectors = segmentStarts - segmentEnds # (L,D)

    distances = np.empty(len(points), dtype=np.double)
    pedalPoints = np.empty_like(points, dtype=np.double)
    closestSegments = np.empty(len(points), dtype=np.intp)
    trimmed = np.zeros(len(points), dtype=np.bool)

    batchCount = max(1, np.ceil(len(points) / maxBatchSize).astype(np.intp))
    for batchPoints, batchDistances, batchPedalPoints, batchClosestSegments, batchTrimmed in zip(
        np.array_split(points, batchCount), np.array_split(distances, batchCount),
        np.array_split(pedalPoints, batchCount), np.array_split(closestSegments, batchCount),
        np.array_split(trimmed, batchCount)
    ):
        pointsToSegmentStarts = segmentStarts - batchPoints[:,np.newaxis] # (N,L,D)
        ## TODO: compare runtime to cupy/torch vecdot
        #tMatrix = xp.einsum('NLD,LD->NL', pointsToSegmentStarts, segmentVectors) / xp.einsum('LD,LD->L', segmentVectors, segmentVectors) # (N,L)
        tMatrix = np.vecdot(pointsToSegmentStarts, segmentVectors) / np.vecdot(segmentVectors, segmentVectors) # (N,L)
        del pointsToSegmentStarts
        potentialPedalPoints = segmentStarts[xp.newaxis] - xp.clip(tMatrix, 0, 1)[:,:,xp.newaxis] * segmentVectors[xp.newaxis] # (N,L,D)

        segmentDistances: npt.NDArray = xp.linalg.norm(potentialPedalPoints - batchPoints[:,np.newaxis], axis=-1) # (N,L)
        batchClosestSegments[:] = segmentDistances.argmin(axis=-1) # (N)
        batchDistances[:] = np.take_along_axis(segmentDistances, batchClosestSegments[:,np.newaxis], axis=-1).squeeze() # (N)

        if returnPedalInfo:
            batchPedalPoints[:] = np.take_along_axis(potentialPedalPoints, batchClosestSegments[:,np.newaxis,np.newaxis], axis=1).squeeze() # (N,D)
        if trimStart:
            batchTrimmed |= (batchClosestSegments == 0) & (tMatrix[:,0] < 0)
        if trimEnd:
            endIdx = len(segmentVectors) - 1
            batchTrimmed |= (batchClosestSegments == endIdx) & (tMatrix[:,endIdx] > 1)

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


def skewLinesDistance(a1: npt.NDArray[np.double], b1: npt.NDArray[np.double], # 2 points on first line
                      a2: npt.NDArray[np.double], b2: npt.NDArray[np.double] # 2 points on second line
                      ) -> float: # lines are defined by 2 points (a, b) on the line
    line1Vector = a2 - a1
    line2Vector = b2 - b1
    normalVector = np.cross(line1Vector, line2Vector)
    connectionVector = a2 - a1

    normalLength = np.linalg.norm(normalVector)
    if np.isclose(normalLength, 0): # vectors are parallel
        return np.linalg.norm(np.cross(connectionVector, line1Vector)) / np.linalg.norm(line1Vector)
    else:
        return np.abs(np.dot(connectionVector, normalVector)) / normalLength
