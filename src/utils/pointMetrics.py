import sys
from multiprocessing import shared_memory
from multiprocessing import get_context

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from typing import Optional, Tuple


_tlsLookUp: Optional[KDTree] = None
_searchRange: float = 0.0
_tlsSharedMemory: Optional[shared_memory.SharedMemory] = None
MAX_DENSITY_BATCH_SIZE = 64


def prepareDensityWorker(tlsLookUp: KDTree):
    """
    Prepare platform-specific context and initializer args for density workers.

    The returned shared-memory handle is owned by the caller and must be closed
    and unlinked after the worker pool has stopped.
    """
    if sys.platform.startswith("linux"):
        return get_context("fork"), (None, None, None, tlsLookUp), None

    tlsSharedMemory = shared_memory.SharedMemory(create=True, size=max(1, tlsLookUp.data.nbytes))
    try:
        sharedTlsPoints = np.ndarray(tlsLookUp.data.shape, dtype=tlsLookUp.data.dtype,
                                     buffer=tlsSharedMemory.buf)
        sharedTlsPoints[:] = tlsLookUp.data
        del sharedTlsPoints
        initializerArgs = (tlsSharedMemory.name, tlsLookUp.data.shape, tlsLookUp.data.dtype.str, None)
        return get_context("spawn"), initializerArgs, tlsSharedMemory
    except BaseException:
        tlsSharedMemory.close()
        tlsSharedMemory.unlink()
        raise


def _calculateBatchDensityParallel(pool,
                                   workers: int,
                                   points: npt.NDArray[np.double],
                                   searchRange: float,
                                   queryPointsIncludeSelf: bool
                                   ) -> npt.NDArray[np.double]:
    batchSize = min(MAX_DENSITY_BATCH_SIZE, max(1, (len(points) + workers * 4 - 1) // (workers * 4)))
    batches = [points[i:i + batchSize] for i in range(0, len(points), batchSize)]
    tasks = [(batch, searchRange, queryPointsIncludeSelf) for batch in batches]
    densityBatches = pool.map(_calculateDensity, tasks)
    return densityBatches


def setWorkerTLSLookUp(tlsLookUp):
    global _tlsLookUp
    _tlsLookUp = tlsLookUp


def initializeDensityWorker(sharedMemoryName: Optional[str] = None,
                            shape: Optional[Tuple[int, ...]] = None,
                            dtype: Optional[str] = None,
                            inheritedTree: Optional[KDTree] = None):
    global _tlsLookUp, _tlsSharedMemory

    if sharedMemoryName is not None:
        if shape is None or dtype is None:
            raise ValueError("Shape and dtype are required for shared TLS points")
        _tlsSharedMemory = shared_memory.SharedMemory(name=sharedMemoryName)
        tlsPoints = np.ndarray(shape, dtype=np.dtype(dtype), buffer=_tlsSharedMemory.buf)
        _tlsLookUp = KDTree(tlsPoints)
    elif inheritedTree is not None:
        _tlsLookUp = inheritedTree
    else:
        raise ValueError("A shared-memory name or inherited TLS KDTree is required")


def _calculateDensity(task: Tuple[npt.NDArray[np.double], float, bool], workers: int = 1) -> npt.NDArray[np.double]:
    """Calculate densities for a batch of points with one vectorized KDTree query."""
    points, searchRange, queryPointsIncludeSelf = task
    points = np.asarray(points, dtype=np.double)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    neighborsByPoint = _tlsLookUp.query_ball_point(points, searchRange, workers=workers)
    densities = np.ones(len(points), dtype=np.double)
    for index, (point, neighbors) in enumerate(zip(points, neighborsByPoint)):
        if queryPointsIncludeSelf and len(neighbors) <= 1:
            continue
        distances: npt.NDArray = np.linalg.norm(_tlsLookUp.data[neighbors] - point, axis=-1)
        if queryPointsIncludeSelf:
            distances = distances[distances > 0]
        weights = decayingWeights(distances, searchRange)
        densities[index] = 1 / (1 + weights.sum())
    return densities


def calculateDensity(points: Optional[npt.NDArray[np.double]],
                     tlsLookUp: KDTree,
                     searchRange: float,
                     workers: int = 1,
                     useMultiProcessing: bool = True,
                     queryPointsIncludeSelf: bool = True
                     ) -> npt.NDArray[np.double]:
    if points is None:
        points = tlsLookUp.data
    if len(points) == 0:
        return np.empty(0, dtype=np.double)

    global _tlsLookUp, _searchRange
    _tlsLookUp, _searchRange = tlsLookUp, searchRange

    if workers > 1 and useMultiProcessing:
        context, initializerArgs, tlsSharedMemory = prepareDensityWorker(tlsLookUp)
        pool = None
        try:
            pool = context.Pool(processes=workers, initializer=initializeDensityWorker, initargs=initializerArgs)
            densityBatches = _calculateBatchDensityParallel(pool, workers, points, searchRange, queryPointsIncludeSelf)
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        finally:
            if pool is not None and pool._state == "RUN":
                pool.close()
                pool.join()
            if tlsSharedMemory is not None:
                tlsSharedMemory.close()
                tlsSharedMemory.unlink()
    else:
        densityBatches = []
        for start in range(0, len(points), MAX_DENSITY_BATCH_SIZE):
            batch = points[start:start + MAX_DENSITY_BATCH_SIZE]
            densityBatches.append(_calculateDensity((batch, searchRange, queryPointsIncludeSelf), workers))

    densities = np.concatenate(densityBatches, axis=0)
    return densities


def decayingWeights(neighborDistances: npt.NDArray[np.double],
                    searchRange: float
                    ) -> npt.NDArray[np.double]:
    # equivalent to the formula  e^{-||x_i, p_j||^2 / (r/2)^2}
    return np.exp(-np.square((2/searchRange) * neighborDistances))
