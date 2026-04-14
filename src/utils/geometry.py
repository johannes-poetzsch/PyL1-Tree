import cupy as cp
import numpy as np
import numpy.typing as npt
from scipy.stats import ttest_ind

from typing import Optional, Tuple


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


def getStripAngles(strip: npt.NDArray[np.double]) -> Optional[npt.NDArray[np.double]]:
    if len(strip) < 3:
        return None
    directions = normalizeVectors(strip[1:] - strip[:-1])
    angles = angle(directions[:-1], directions[1:])
    return angles


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


def movingTTest(sequence: npt.NDArray[np.double]) -> Optional[int]:
    windowSize = max(5, int(np.floor(len(sequence) / 10)))

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


def stripPointsDistances(strip: npt.NDArray, # (L+1,D)
                         points: npt.NDArray, # (N,D)
                         trimStart = False,
                         trimEnd = False,
                         returnTrimming = False,
                         returnPedalInfo = False
                         ) -> (Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.bool]] |
                               Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.intp], npt.NDArray[np.intp]] |
                               Tuple[npt.NDArray[np.double], npt.NDArray[np.bool]] |
                               npt.NDArray[np.double]):
    xp = cp.get_array_module(strip, points)
    segmentStarts = strip[:-1]
    segmentEnds = strip[1:]

    segmentVectors = segmentStarts - segmentEnds # (L,D)
    pointsToSegmentStarts = segmentStarts[:,xp.newaxis] - points # (L,N,D)
    # TODO: compare runtime to torch vecdot
    tMatrix = xp.einsum('LND,LD->LN', pointsToSegmentStarts, segmentVectors) / xp.einsum('LD,LD->L', segmentVectors, segmentVectors)[:,xp.newaxis] # (L,N)
    del pointsToSegmentStarts
    xp.clip(tMatrix, 0, 1, out=tMatrix)
    pedalPoints = segmentStarts[:,xp.newaxis] - tMatrix[:,:,xp.newaxis] * segmentVectors[:,xp.newaxis] # (L,N,D)

    segmentDistances: npt.NDArray = xp.linalg.norm(pedalPoints - points, axis=-1) # (L,N)
    closestSegment = segmentDistances.argmin(axis=0) # (N)
    distances: npt.NDArray[np.double] = np.take_along_axis(segmentDistances, closestSegment[np.newaxis], axis=0).squeeze(0)

    trimmed = np.zeros_like(distances, dtype=bool)

    if trimStart:
        trimmed |= (closestSegment == 0) & (tMatrix[0] == 0)
    if trimEnd:
        endIdx = len(segmentVectors) - 1
        trimmed |= (closestSegment == endIdx) & (tMatrix[endIdx] == 1)

    untrimmed = ~trimmed
    if returnPedalInfo:
        pedalPoints = np.take_along_axis(pedalPoints, np.expand_dims(closestSegment, axis=(0,2)), axis=0).squeeze(0)
        atStartPoint = np.take_along_axis(tMatrix, closestSegment[np.newaxis], axis=0).squeeze(0) < 0.5
        closestPointIndices = np.where(atStartPoint, closestSegment, closestSegment+1)
        if returnTrimming:
            return distances, pedalPoints, closestSegment, closestPointIndices, trimmed
        else:
            return (distances[untrimmed], pedalPoints[untrimmed],
                    closestSegment[untrimmed], closestPointIndices[untrimmed])
    else:
        if returnTrimming:
            return distances, trimmed
        else:
            return distances[untrimmed]
