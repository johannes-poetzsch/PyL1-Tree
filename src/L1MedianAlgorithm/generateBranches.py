import sys
from multiprocessing import get_context, shared_memory
from multiprocessing.pool import Pool
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Any, Dict, List, Optional, Tuple

from .findBranches import findNewBranches
from .runIteration import (MAX_QUERY_BATCH_SIZE, initializeWorker, runIteration,
                           runIterationBatch, runIterationBatchesLocally, setIterationParameters)
from ..objects.branch import Branch
from ..utils.pointMetrics import calculateDensity


def generateBranches(tlsPoints: npt.NDArray[np.double],
                     seedPoints: npt.NDArray[np.double],
                     relPathLocations: npt.NDArray[np.double],
                     parameters: Dict[str, Any],
                     workers: int = 1,
                     useMultiProcessing: bool = True
                     ) -> Tuple[List[Branch], npt.NDArray[np.double]]:
    tlsLookUp = KDTree(tlsPoints)
    skeletonData = initializeSkeleton(seedPoints, relPathLocations)
    pool = None
    tlsSharedMemory = None
    skeletonSharedMemory = None
    densitiesSharedMemory = None
    workerSkeletonData = None
    workerTlsDensities = None
    if workers > 1 and useMultiProcessing:
        skeletonSharedMemory = shared_memory.SharedMemory(create=True, size=max(1, skeletonData.nbytes))
        densitiesSharedMemory = shared_memory.SharedMemory(create=True,
                                                           size=max(1, tlsPoints.shape[0] * np.dtype(np.double).itemsize))
        workerSkeletonData = np.ndarray(skeletonData.shape, dtype=skeletonData.dtype,
                                        buffer=skeletonSharedMemory.buf)
        workerTlsDensities = np.ndarray((tlsPoints.shape[0],), dtype=np.double,
                                        buffer=densitiesSharedMemory.buf)

        if sys.platform.startswith("linux"):
            # Fork inherits the parent's read-only tree through copy on write.
            initializeWorker(inheritedTree=tlsLookUp)
            pool = get_context("fork").Pool(processes=workers, initializer=initializeWorker,
                                            initargs=(None, None, None, skeletonSharedMemory.name,
                                                      skeletonData.shape, skeletonData.dtype,
                                                      densitiesSharedMemory.name, workerTlsDensities.shape,
                                                      workerTlsDensities.dtype.str))
        else:
            tlsSharedMemory = shared_memory.SharedMemory(create=True, size=tlsLookUp.data.nbytes)
            sharedTlsPoints = np.ndarray(tlsLookUp.data.shape, dtype=tlsLookUp.data.dtype,
                                         buffer=tlsSharedMemory.buf)
            sharedTlsPoints[:] = tlsLookUp.data
            pool = get_context("spawn").Pool(processes=workers, initializer=initializeWorker,
                                             initargs=(tlsSharedMemory.name, tlsLookUp.data.shape,
                                                       tlsLookUp.data.dtype.str, skeletonSharedMemory.name,
                                                       skeletonData.shape, skeletonData.dtype,
                                                       densitiesSharedMemory.name, workerTlsDensities.shape,
                                                       workerTlsDensities.dtype.str))
    try:
        tlsDensities = calculateDensity(None, tlsLookUp, 2 * parameters["initialSearchRange"], workers, pool,
                                        useMultiProcessing)
        if workerTlsDensities is not None:
            workerTlsDensities[:] = tlsDensities
        return _generateBranchesWithPool(tlsLookUp, skeletonData, tlsDensities, parameters, workers, pool,
                                         workerSkeletonData, useMultiProcessing)
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
        if skeletonSharedMemory is not None:
            skeletonSharedMemory.close()
            skeletonSharedMemory.unlink()
        if densitiesSharedMemory is not None:
            densitiesSharedMemory.close()
            densitiesSharedMemory.unlink()


def _generateBranchesWithPool(tlsLookUp, skeletonData, tlsDensities, parameters, workers, pool,
                              workerSkeletonData, useMultiProcessing):
    skeletonBranches: List[Branch] = []
    searchRange = parameters["searchRange"]
    maxSearchRange = 2 * parameters["baseRadius"]
    iterationId = 0

    while searchRange <= maxSearchRange:
        #print(searchRange, " / ", maxSearchRange)
        for _ in range(parameters["maxIterationNumFixedRange"]):
            recomputeSkeletonPoints(skeletonData, tlsLookUp, tlsDensities, searchRange,
                                    parameters["baseRadius"], parameters["minMu"], parameters["maxMu"], workers,
                                    pool=pool, workerSkeletonData=workerSkeletonData,
                                    iterationId=iterationId,
                                    useMultiProcessing=useMultiProcessing)
            iterationId += 1

        skeletonData = filterCloseDataPoints(skeletonData, parameters["nearThreshold"])
        computeSmoothDirectionalDegrees(skeletonData, parameters["kSmoothSigma"],
                                        workers if not useMultiProcessing else 1)
        setFixPoints(skeletonData, parameters["sigmaThreshold"])
        skeletonBranches.extend(findNewBranches(skeletonData,
                                                parameters["kTracingThreshold"],
                                                parameters["distanceTracingThreshold"],
                                                np.deg2rad(parameters["angleTracingThreshold"]),
                                                parameters["sigmaTracingThreshold"],
                                                parameters["branchSizeThreshold"],
                                                workers if not useMultiProcessing else 1))

        searchRange *= 1 + parameters["iterationSpeed"]

    unfixedPoints = skeletonData["Coords"][~skeletonData["isFix"]]

    return skeletonBranches, unfixedPoints


def initializeSkeleton(seedPoints: npt.NDArray[np.double],
                       relPathLocations: npt.NDArray[np.double]) -> npt.NDArray:
    skeletonData = np.zeros(len(seedPoints),
                            dtype=[("Coords", (np.double, 3)),
                                   ("relPathLocation", np.double),
                                   ("Sigma", np.double),
                                   ("isFix", np.bool_),
                                   ("isBranch", np.bool_)])
    skeletonData["Coords"] = seedPoints
    skeletonData["relPathLocation"] = relPathLocations

    return skeletonData


def recomputeSkeletonPoints(skeletonData: npt.NDArray,
                            tlsLookUp: KDTree,
                            tlsDensities: npt.NDArray[np.double],
                            searchRange: float,
                            baseRadius: float,
                            minMu: float,
                            maxMu: float,
                            workers: int = 1,
                            pool: Optional[Pool] = None,
                            workerSkeletonData: Optional[npt.NDArray] = None,
                            iterationId: int = 0,
                            useMultiProcessing: bool = True
                            ) -> Optional[Tuple[npt.NDArray, npt.NDArray]]:
    sigma = skeletonData["Sigma"]
    changing = ~skeletonData["isFix"]
    changingIndices = np.flatnonzero(changing)


    if workers > 1:
        if useMultiProcessing and pool is not None:
            workerSkeletonData[:len(skeletonData)] = skeletonData
            batchSize = min(MAX_QUERY_BATCH_SIZE,
                            max(1, (len(changingIndices) + workers * 4 - 1) // (workers * 4)))
            batches = [changingIndices[i:i + batchSize] for i in range(0, len(changingIndices), batchSize)]
            tasks = [(batch, baseRadius, searchRange, len(skeletonData), iterationId) for batch in batches]
            results = (result for batchResults in pool.map(runIterationBatch, tasks) for result in batchResults)
        else:
            # run Python code on a single thread, but pass the workers argument to KDTree queries
            results = iter(runIterationBatchesLocally(changingIndices, baseRadius, searchRange, skeletonData,
                                                      tlsDensities, tlsLookUp, MAX_QUERY_BATCH_SIZE, workers))
    else:
        setIterationParameters(baseRadius, searchRange, skeletonData, tlsDensities, tlsLookUp)
        results = map(runIteration, changingIndices)

    repulsion = np.zeros((len(changingIndices), 3), dtype=np.double)
    tlsAverage = np.zeros((len(changingIndices), 3), dtype=np.double)
    updated = np.zeros(len(changingIndices), dtype=np.bool_)

    for resultIndex, (idx, canChange, pointSigma, pointRepulsion, pointAverage) in enumerate(results):
        changing[idx] = canChange
        sigma[idx] = pointSigma
        if canChange:
            updated[resultIndex] = True
            repulsion[resultIndex] = pointRepulsion
            tlsAverage[resultIndex] = pointAverage

    minSigma = sigma.min()
    maxSigma = sigma.max()
    sigmaRange = maxSigma - minSigma

    muRange = np.abs(maxMu - minMu)
    mu = (muRange / sigmaRange) * (sigma - minSigma) + minMu

    changing[mu < 0] = False
    compactChanging = updated & changing[changingIndices]
    updatedIndices = changingIndices[compactChanging]

    skeletonData["Coords"][updatedIndices] = (
            tlsAverage[compactChanging] + mu[updatedIndices, np.newaxis] * repulsion[compactChanging])


def filterCloseDataPoints(skeletonData: npt.NDArray,
                          nearThreshold: float
                          ) -> npt.NDArray:
    kept = np.ones_like(skeletonData, dtype=bool)

    skeletonLookUp = KDTree(skeletonData["Coords"])
    skeletonNeighbors = skeletonLookUp.query_ball_tree(skeletonLookUp, nearThreshold)
    for idx in range(len(skeletonData)):
        if not kept[idx] or len(skeletonNeighbors[idx]) <= 1:
            continue

        neighbors = np.array(skeletonNeighbors[idx])
        changing = ~skeletonData["isFix"][neighbors]
        kept[neighbors[changing]] = False
        kept[idx] = True

    return skeletonData[kept]


def computeSmoothDirectionalDegrees(skeletonData: npt.NDArray,
                                    kNeighbors: int,
                                    workers: int = 1):
    skeletonLookUp = KDTree(skeletonData["Coords"])
    notFix = ~skeletonData["isFix"] # (n)
    notFixIndices = np.flatnonzero(notFix)

    if len(notFixIndices) == 0:
        return

    _, neighbors = skeletonLookUp.query(skeletonData[notFix]["Coords"], kNeighbors + 1,
                                        workers=workers) # (nF,k+1)
    neighbors = neighbors[:,1:] # (nF,k)

    if np.any(neighbors == len(skeletonData)):
        raise ValueError("Found too few neighbors to calculate smooth directional degrees")

    neighborIsValid = notFix[neighbors] # (nF,k)
    validNeighborsCount = np.count_nonzero(neighborIsValid, axis=1) # (nF)

    hasEnoughValidNeighbors = validNeighborsCount > 3 # (nF)
    sigma = np.full(len(notFixIndices), 0.95, dtype=np.double)

    validRows = np.flatnonzero(hasEnoughValidNeighbors)
    if len(validRows) > 0:
        neighborVectors = (
            skeletonData["Coords"][neighbors[validRows]] - skeletonData["Coords"][notFixIndices[validRows], np.newaxis])
        neighborVectors[~neighborIsValid[validRows]] = 0.0

        covariance = np.einsum("mki,mkj->mij", neighborVectors, neighborVectors)
        eigenValues = np.linalg.eigvalsh(covariance)
        sigma[validRows] = eigenValues[:, -1] / eigenValues.sum(axis=1)

    skeletonData["Sigma"][notFixIndices] = sigma

    invSigma = 1.0 - skeletonData["Sigma"] # (n)

    neighborInvSigmaSum = np.where(neighborIsValid, invSigma[neighbors], 0.0).sum(axis=-1) # (nF)
    sigma = 1.0 - ((invSigma[notFix] + neighborInvSigmaSum) / (1.0 + validNeighborsCount))

    skeletonData["Sigma"][notFix] = np.where(sigma < 0, 0.5, sigma)


def setFixPoints(skeletonData: npt.NDArray,
                 sigmaThreshold: float):
    isBranch = skeletonData["isBranch"]
    skeletonData["isFix"][~isBranch] = skeletonData["Sigma"][~isBranch] > sigmaThreshold
