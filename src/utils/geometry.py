import cupy as cp
import numpy as np
import numpy.typing as npt


def normalizeVectors(vectors: npt.NDArray):
    xp = cp.get_array_module(vectors)
    vectors /= xp.expand_dims(xp.linalg.norm(vectors, axis=-1), axis=-1)


def linesPointsDistances(lineStartPoints: npt.NDArray, # (L,D)
                         lineEndPoints: npt.NDArray, # (L,D) or (D)
                         points: npt.NDArray) -> npt.NDArray: # (N,D)
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

    return distances


def stripPointsDistances(strip: npt.NDArray, # (L+1,D)
                         points: npt.NDArray, # (N,D)
                         trimStart = False,
                         trimEnd = False) -> npt.NDArray[np.double]:
    xp = cp.get_array_module(strip, points, points)
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
    distances = segmentDistances.min(axis=0)

    if not (trimStart or trimEnd):
        return distances
    else:
        closestSegment = segmentDistances.argmin(axis=0) # (N)

        endIdx = len(segmentVectors) - 1

        if trimStart and trimEnd:
            trimmed = xp.logical_or((closestSegment == 0) & (tMatrix[0] == 0),
                                    (closestSegment == endIdx) & (tMatrix[endIdx] == 1))
        elif trimStart:
            trimmed = (closestSegment == 0) & (tMatrix[0] == 0)
        else:
            trimmed = (closestSegment == endIdx) & (tMatrix[endIdx] == 1)

        return distances[xp.logical_not(trimmed)]


def angle(directions_a: npt.NDArray,
          directions_b: npt.NDArray,
          vectorsAreNormalized: bool = True) -> npt.NDArray:
    # TODO: compare to torch vecdot
    if not vectorsAreNormalized:
        normalizeVectors(directions_a)
        normalizeVectors(directions_b)
    cos = np.vecdot(directions_a, directions_b)
    np.clip(cos, -1, 1, out=cos)
    return np.arccos(cos)
