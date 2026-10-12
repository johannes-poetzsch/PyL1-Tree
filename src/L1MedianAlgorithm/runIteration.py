from multiprocessing import shared_memory
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Any, List, Optional, Tuple

from src.utils.pointMetrics import decayingWeights, setWorkerTLSLookUp


_baseRadius: Optional[float] = None
_searchRange: Optional[float] = None
_skeletonData: Optional[npt.NDArray] = None
_tlsDensities: Optional[npt.NDArray[np.double]] = None
_skeletonLookUp: Optional[KDTree] = None
_tlsLookUp: Optional[KDTree] = None
_tlsSharedMemory = None
_skeletonSharedMemory = None
_densitiesSharedMemory = None
_workerSkeletonBuffer = None
_workerDensitiesBuffer = None
_workerIterationId = None
_workerIterationTree: Optional[KDTree] = None
MAX_QUERY_BATCH_SIZE = 64


def initializeWorker(sharedMemoryName: Optional[str] = None,
                     shape: Optional[Tuple[Any, ...]] = None,
                     dtype: Optional[np.dtype] = None,
                     skeletonMemoryName: Optional[str] = None,
                     skeletonShape: Optional[Tuple[Any, ...]] = None,
                     skeletonDtype: Optional[np.dtype] = None,
                     densitiesMemoryName: Optional[Tuple[Any, ...]] = None,
                     densitiesShape: Optional[Tuple[Any, ...]] = None,
                     densitiesDtype: Optional[np.dtype] = None,
                     inheritedTree: Optional[KDTree] = None):
    """Attach workers to shared inputs; build the read-only TLS tree once."""
    global _tlsLookUp, _tlsSharedMemory, _skeletonSharedMemory, _densitiesSharedMemory
    global _workerSkeletonBuffer, _workerDensitiesBuffer
    if sharedMemoryName is not None:
        _tlsSharedMemory = shared_memory.SharedMemory(name=sharedMemoryName)
        tlsPoints = np.ndarray(shape, dtype=np.dtype(dtype), buffer=_tlsSharedMemory.buf)
        _tlsLookUp = KDTree(tlsPoints)
    elif inheritedTree is not None:
        _tlsLookUp = inheritedTree
    if skeletonMemoryName is not None:
        _skeletonSharedMemory = shared_memory.SharedMemory(name=skeletonMemoryName)
        _workerSkeletonBuffer = np.ndarray(skeletonShape, dtype=np.dtype(skeletonDtype),
                                           buffer=_skeletonSharedMemory.buf)
    if densitiesMemoryName is not None:
        _densitiesSharedMemory = shared_memory.SharedMemory(name=densitiesMemoryName)
        _workerDensitiesBuffer = np.ndarray(densitiesShape, dtype=np.dtype(densitiesDtype),
                                            buffer=_densitiesSharedMemory.buf)
    setWorkerTLSLookUp(_tlsLookUp)


def setIterationParameters(baseRadius: float,
                           searchRange: float,
                           skeletonData: npt.NDArray,
                           tlsDensities: npt.NDArray[np.double],
                           tlsLookUp: KDTree):
    global _baseRadius, _searchRange, _skeletonData, _tlsDensities, _skeletonLookUp, _tlsLookUp

    _baseRadius = baseRadius
    _searchRange = searchRange
    _skeletonData = skeletonData
    _tlsDensities = tlsDensities
    _skeletonLookUp = KDTree(skeletonData["Coords"])
    _tlsLookUp = tlsLookUp


def runIteration(idx: int) -> Tuple[int, bool, np.double, Optional[npt.NDArray[np.double]], Optional[npt.NDArray[np.double]]]:
    neighborRadius = np.maximum(2 * _searchRange, 2 * _baseRadius * _skeletonData["relPathPosition"][idx])
    neighbors = _skeletonLookUp.query_ball_point(_skeletonData["Coords"][idx], neighborRadius)

    if len(neighbors) <= 1:
        return idx, False, 0.5, None, None

    neighbors = [n for n in neighbors if n != idx]
    neighborVectors = _skeletonData["Coords"][idx] - _skeletonData["Coords"][neighbors]
    neighborDistances = np.linalg.norm(neighborVectors, axis=-1)
    _sigma = calculateDirectionalDegree(neighborVectors, neighborDistances, _searchRange)
    _repulsion = calculateRepulsion(neighborVectors, neighborDistances, _searchRange)

    tlsNeighbors = _tlsLookUp.query_ball_point(_skeletonData["Coords"][idx], neighborRadius)
    if len(tlsNeighbors) == 0:
        return idx, False, _sigma, None, None

    _tlsAverage = calculatePointAverage(_skeletonData["Coords"][idx], _tlsLookUp.data[tlsNeighbors],
                                        _tlsDensities[tlsNeighbors], _searchRange)
    return idx, True, _sigma, _repulsion, _tlsAverage


def _runIterationBatch(indices: npt.NDArray[np.intp], queryWorkers: int
                       ) -> List[Tuple[int, bool, np.double, Optional[npt.NDArray[np.double]], Optional[npt.NDArray[np.double]]]]:
    """Query both trees for a bounded batch, then calculate each point's update."""
    if len(indices) == 0:
        return []

    points = _skeletonData["Coords"][indices]
    radii = np.maximum(2 * _searchRange,
                       2 * _baseRadius * _skeletonData["relPathPosition"][indices])
    skeletonNeighbors = _skeletonLookUp.query_ball_point(points, radii, workers=queryWorkers,
                                                         return_sorted=False)
    tlsNeighbors = _tlsLookUp.query_ball_point(points, radii, workers=queryWorkers,
                                               return_sorted=False)

    results = []
    for resultIndex, idx in enumerate(indices):
        neighbors = skeletonNeighbors[resultIndex]
        if len(neighbors) <= 1:
            results.append((idx, False, 0.5, None, None))
            continue

        neighbors = [neighbor for neighbor in neighbors if neighbor != idx]
        neighborVectors = _skeletonData["Coords"][idx] - _skeletonData["Coords"][neighbors]
        neighborDistances = np.linalg.norm(neighborVectors, axis=-1)
        sigma = calculateDirectionalDegree(neighborVectors, neighborDistances, _searchRange)
        repulsion = calculateRepulsion(neighborVectors, neighborDistances, _searchRange)

        pointTlsNeighbors = tlsNeighbors[resultIndex]
        if len(pointTlsNeighbors) == 0:
            results.append((idx, False, sigma, None, None))
            continue

        tlsAverage = calculatePointAverage(_skeletonData["Coords"][idx],
                                           _tlsLookUp.data[pointTlsNeighbors],
                                           _tlsDensities[pointTlsNeighbors], _searchRange)
        results.append((idx, True, sigma, repulsion, tlsAverage))
    return results


def runIterationBatch(task: Tuple[npt.NDArray[np.intp], float, float, int, int]
                      ) -> List[Tuple[int, bool, np.double, Optional[npt.NDArray[np.double]], Optional[npt.NDArray[np.double]]]]:
    """Process-pool entry point; query threading stays disabled per process."""
    global _baseRadius, _searchRange, _skeletonData, _tlsDensities, _skeletonLookUp
    global _workerIterationId, _workerIterationTree
    indices, baseRadius, searchRange, skeletonCount, iterationId = task
    if iterationId != _workerIterationId:
        _skeletonData = _workerSkeletonBuffer[:skeletonCount]
        _tlsDensities = _workerDensitiesBuffer
        _workerIterationTree = KDTree(_skeletonData["Coords"])
        _workerIterationId = iterationId
    _baseRadius = baseRadius
    _searchRange = searchRange
    _skeletonLookUp = _workerIterationTree
    return _runIterationBatch(indices, queryWorkers=1)


def runIterationBatchesLocally(indices: npt.NDArray[np.intp],
                               baseRadius: float,
                               searchRange: float,
                               skeletonData: npt.NDArray,
                               tlsDensities: npt.NDArray[np.double],
                               tlsLookUp: KDTree,
                               batchSize: int,
                               queryWorkers: int
                               ) -> List[Tuple[int, bool, np.double, Optional[npt.NDArray[np.double]], Optional[npt.NDArray[np.double]]]]:
    """Run bounded batches in this process using SciPy's query parallelism."""
    setIterationParameters(baseRadius, searchRange, skeletonData, tlsDensities, tlsLookUp)
    results = []
    for start in range(0, len(indices), batchSize):
        results.extend(_runIterationBatch(indices[start:start + batchSize], queryWorkers))
    return results


def calculateDirectionalDegree(neighborVectors: npt.NDArray[np.double],
                               neighborDistances: Optional[npt.NDArray[np.double]] = None,
                               searchRange: Optional[float] = None,
                               useDecayingWeights: bool = True) -> np.double:
    if len(neighborVectors) <= 3:
        return np.double(0.5)

    if useDecayingWeights:
        assert(searchRange is not None and neighborDistances is not None)
        weights = decayingWeights(neighborDistances, 2. * searchRange)
    else:
        weights = np.ones(1, dtype=np.double)
    
    c = neighborVectors.T @ (neighborVectors * weights[:, None])
    eigenValues = np.linalg.eigvalsh(c)
    sigma = eigenValues[-1] / eigenValues.sum()
    return sigma


def calculateRepulsion(neighborVectors: npt.NDArray[np.double],
                       neighborDistances: npt.NDArray[np.double],
                       searchRange: float
                       ) -> npt.NDArray[np.double]:
    neighborDistances = np.clip(neighborDistances, a_min=0.001 * searchRange, a_max=None)

    weights = decayingWeights(neighborDistances, 2. * searchRange)
    weights /= neighborDistances

    repulsionTerm = weights.transpose() @ neighborVectors

    return repulsionTerm / weights.sum()


def calculatePointAverage(point: npt.NDArray[np.double],
                          neighborPoints: npt.NDArray[np.double],
                          densities: npt.NDArray[np.double],
                          searchRange: float
                          ) -> npt.NDArray[np.double]:
    distances = np.linalg.norm(neighborPoints - point, axis=-1)
    np.clip(distances, a_min=0.001 * searchRange, a_max=None, out=distances)

    weights = decayingWeights(distances, 2 * searchRange)
    weights *= densities

    averageTerm = weights.transpose() @ neighborPoints

    return averageTerm / weights.sum()
