import cupy as cp
import numpy as np
import numpy.typing as npt

from typing import Optional, Tuple


def distanceAlongDirection(startPoints: npt.NDArray[np.double],
                           endPoints: npt.NDArray[np.double],
                           direction: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    return np.vecdot(endPoints - startPoints, direction)


def normalizeVectors(vectors: npt.NDArray):
    xp = cp.get_array_module(vectors)
    return vectors / xp.expand_dims(xp.linalg.norm(vectors, axis=-1), axis=-1)


def linesPointsDistances(lineStartPoints: npt.NDArray, # (L,D)
                         lineEndPoints: npt.NDArray, # (L,D) or (D)
                         points: npt.NDArray, # (N,D)
                         returnPedalPoints: bool = False) -> npt.NDArray | Tuple[npt.NDArray, npt.NDArray]:
    xp = cp.get_array_module(lineStartPoints, lineEndPoints, points)
    lineVectors = lineStartPoints - lineEndPoints # (L,D)

    pointsToLineStartPoints = lineStartPoints[:,xp.newaxis] - points # (L,N,D)
    # TODO: compare runtime to torch vecdot
    tMatrix = xp.einsum('LND,LD->LN', pointsToLineStartPoints, lineVectors) / xp.einsum('LD,LD->L', lineVectors, lineVectors)[:,xp.newaxis] # (L,N)
    del pointsToLineStartPoints
    xp.clip(tMatrix, 0, 1, out=tMatrix)
    closestPointsOnLines = lineStartPoints[:,xp.newaxis] - tMatrix[:,:,xp.newaxis] * lineVectors[:,xp.newaxis] # (L,N,D)
    del tMatrix
    distances = xp.linalg.norm(closestPointsOnLines - points, axis=-1) # (L,N)

    if returnPedalPoints:
        return distances, closestPointsOnLines
    else:
        return distances


def stripPointsDistances(strip: npt.NDArray, # (L+1,D)
                         points: npt.NDArray, # (N,D)
                         trimStart = False,
                         trimEnd = False,
                         returnTrimming = False
                         ) -> (npt.NDArray[np.double] | Tuple[npt.NDArray[np.double], npt.NDArray[np.bool]]):
    xp = cp.get_array_module(strip, points)
    segmentStarts = strip[:-1]
    segmentEnds = strip[1:]

    segmentVectors = segmentStarts - segmentEnds # (L,D)
    pointsToSegmentStarts = segmentStarts[:,xp.newaxis] - points # (L,N,D)
    # TODO: compare runtime to torch vecdot
    tMatrix = xp.einsum('LND,LD->LN', pointsToSegmentStarts, segmentVectors) / xp.einsum('LD,LD->L', segmentVectors, segmentVectors)[:,xp.newaxis] # (L,N)
    del pointsToSegmentStarts
    xp.clip(tMatrix, 0, 1, out=tMatrix)
    closestSegmentPoints = segmentStarts[:,xp.newaxis] - tMatrix[:,:,xp.newaxis] * segmentVectors[:,xp.newaxis] # (L,N,D)

    segmentDistances: npt.NDArray = xp.linalg.norm(closestSegmentPoints - points, axis=-1) # (L,N)
    distances: npt.NDArray[np.double] = segmentDistances.min(axis=0)

    trimmed = np.zeros_like(distances, dtype=bool)
    closestSegment = segmentDistances.argmin(axis=0) # (N)

    if trimStart:
        trimmed |= (closestSegment == 0) & (tMatrix[0] == 0)
    if trimEnd:
        endIdx = len(segmentVectors) - 1
        trimmed |= (closestSegment == endIdx) & (tMatrix[endIdx] == 1)

    if returnTrimming:
        return distances, trimmed
    else:
        return distances[xp.logical_not(trimmed)]
        

def getStripAngles(strip: npt.NDArray[np.double]) -> Optional[npt.NDArray[np.double]]:
    if len(strip) < 3:
        return None
    directions = normalizeVectors(strip[1:] - strip[:-1])
    angles = angle(directions[:-1], directions[1:])
    return angles


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
