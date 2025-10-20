import cupy as cp
import numpy as np
import numpy.typing as npt


def normalizeVectors(vectors: npt.NDArray):
    xp = cp.get_array_module(vectors)
    axis = vectors.ndim - 1
    vectors /= xp.expand_dims(xp.linalg.norm(vectors, axis=axis), axis=axis)


def linesPointsDistances(lineStartPoints: npt.NDArray, # (L,D) or (D)
                         lineEndPoints: npt.NDArray, # (L,D)
                         points: npt.NDArray) -> npt.NDArray: # (N,D)
    xp = cp.get_array_module(lineStartPoints, lineEndPoints, points)
    lineVectors = lineEndPoints - lineStartPoints # (L,D)

    pointsToLineEndPoints = lineEndPoints[:,xp.newaxis] - points # (L,N,D)
    # TODO: compare runtime to torch vecdot
    tMatrix = xp.einsum('LND,LD->LN', pointsToLineEndPoints, lineVectors) / xp.einsum('LD,LD->L', lineVectors, lineVectors)[:,xp.newaxis] # (L,N)
    del pointsToLineEndPoints
    xp.clip(tMatrix, 0, 1, out=tMatrix)
    closestPointsOnLines = lineEndPoints[:,xp.newaxis] - tMatrix[:,:,xp.newaxis] * lineVectors[:,xp.newaxis] # (L,N,D)
    del tMatrix
    distances = xp.linalg.norm(closestPointsOnLines - points, axis=-1) # (L,N)

    return distances


def angle(directions_a: npt.NDArray,
          directions_b: npt.NDArray) -> npt.NDArray:
    # TODO: compare to torch vecdot
    cos = np.vecdot(directions_a, directions_b)
    np.clip(cos, -1, 1, out=cos)
    return np.arccos(cos)    
