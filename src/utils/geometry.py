import cupy as cp
import numpy as np
import numpy.typing as npt


def normalizeVectors(vectors: npt.NDArray):
    xp = cp.get_array_module(vectors)
    axis = vectors.ndim - 1
    vectors /= xp.expand_dims(xp.linalg.norm(vectors, axis=axis), axis=axis)


def vecdot(a: npt.NDArray,
           b: npt.NDArray) -> npt.NDArray:
    xp = cp.get_array_module(a, b)
    if b.ndim > a.ndim:
        a, b = b, a
    if b.ndim == 1:
        return xp.dot(a,b)
    elif b.ndim == 2:
        return xp.einsum('ij,ij->i', a, b)
    else:
        raise ValueError("Unexpected Number of dimensions for dot product calculation")


def linesPointsDistances(lineStartPoints: npt.NDArray, # (L,D) or (D)
                         lineEndPoints: npt.NDArray, # (L,D)
                         points: npt.NDArray) -> npt.NDArray: # (N,D)
    xp = cp.get_array_module(lineStartPoints, lineEndPoints, points)
    lineVectors = lineEndPoints - lineStartPoints # (L,D)

    pointsToLineEndPoints = lineEndPoints[:,xp.newaxis] - points # (L,N,D)
    tMatrix = xp.einsum('LND,LD->LN', pointsToLineEndPoints, lineVectors) / xp.einsum('LD,LD->L', lineVectors, lineVectors)[:,xp.newaxis] # (L,N)
    del pointsToLineEndPoints
    xp.clip(tMatrix, 0, 1, out=tMatrix)
    closestPointsOnLines = lineEndPoints[:,xp.newaxis] - tMatrix[:,:,xp.newaxis] * lineVectors[:,xp.newaxis] # (L,N,D)
    del tMatrix
    distances = xp.linalg.norm(closestPointsOnLines - points, axis=-1) # (L,N)

    return distances


def angle(directions_a: npt.NDArray,
          directions_b: npt.NDArray) -> npt.NDArray:
    cos = np.vecdot(directions_a, directions_b)
    cos = np.clip(cos, -1, 1)
    return np.arccos(cos)    
