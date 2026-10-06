import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from typing import Any, Dict, Optional

from src.filterBranches import limitToRange
from src.objects.branch import Branch
from src.utils.branchUtils import generateSearchRanges
from src.utils.geometry import distanceAlongDirection, getStripAngles, normalizeVectors, stripPointsDistances
from src.utils.iteration import breadthFirstSearch
from src.utils.pointMetrics import calculateDensity


def smoothBranches(trunk: Branch,
                   tlsPoints: npt.NDArray[np.double],
                   unassignedTlsPoints: npt.NDArray[np.double],
                   parameters: Dict[str, Any],
                   workers: int = 1,
                   useMultiprocessing: bool = True,
                   ) -> npt.NDArray[np.double]:
    cleanUpBranches(trunk, unassignedTlsPoints, includeNewPointsForRanges=False)

    tlsLookUp = KDTree(tlsPoints)
    densities = calculateDensity(tlsPoints, tlsLookUp, 2 * parameters["initialSearchRange"], workers,
                                 useMultiprocessing)
    for branch in list(breadthFirstSearch(trunk)):
        smoothBranch(branch, tlsLookUp, 2 * parameters["initialSearchRange"], parameters["resolution"],
                     np.deg2rad(parameters["smoothAngleThreshold"]), densities)

    return unassignedTlsPoints


def cleanUpBranches(trunk: Branch,
                    unassignedPoints: npt.NDArray[np.double],
                    includeNewPointsForRanges: bool = True
                    ) -> npt.NDArray[np.double]:
    removeBranchAnomalies(trunk)

    branches = list(breadthFirstSearch(trunk))

    if includeNewPointsForRanges:
        unassignedTlsPoints = Branch.reassignPoints(branches, unassignedPoints)

    searchRanges = generateSearchRanges(branches, useMedianForStem=False, useTrimming=True)

    if not includeNewPointsForRanges:
        unassignedTlsPoints = Branch.reassignPoints(branches, unassignedPoints)

    unassignedTlsPoints = np.append(unassignedTlsPoints, limitToRange(branches, searchRanges), axis=0)

    return unassignedTlsPoints


def removeBranchAnomalies(trunk):
    rightAngle = np.deg2rad(90)
    for branch in breadthFirstSearch(trunk):
        wasChanged = False
        while True:
            if len(branch) < 3:
                break

            angles = branch.getAngles()
            maxIdx = angles.argmax()

            if angles[maxIdx] >= rightAngle:
                branch.skeletonPoints = np.delete(branch.skeletonPoints, maxIdx + 1, axis=0)
                branch.endPointIndices = None
                wasChanged = True
            else:
                break
        if wasChanged:
            branch.updateSkeletonMeasurements()


def smoothBranch(branch: Branch, 
                 tlsLookUp: KDTree,
                 searchRange: float,
                 resolution: float,
                 smoothAngleThreshold: float,
                 tlsDensities: Optional[npt.NDArray[np.double]] = None,
                 workers: int = 1,
                 useMultiprocessing: bool = True):
    _, trimmed = branch.getDistances(branch.tlsPoints, trimStart=True, trimEnd=True, returnTrimming=True)
    tlsPoints = branch.tlsPoints[~trimmed]
    if len(tlsPoints) < 10:
        return

    if tlsDensities is None:
        densities = calculateDensity(tlsPoints, tlsLookUp, searchRange, workers, useMultiprocessing)
    else:
        tlsIndices = tlsLookUp.query(tlsPoints)[1]
        densities = tlsDensities[tlsIndices]
    weightedPoints = tlsPoints * densities[:, np.newaxis]

    startPoint: npt.NDArray[np.double] = branch.skeletonPoints[0]
    endPoint: npt.NDArray[np.double] = branch.skeletonPoints[-1]

    direction = normalizeVectors(endPoint - startPoint)

    distanceToStart = distanceAlongDirection(startPoint, tlsPoints, direction)
    order = distanceToStart.argsort()

    branchLength = np.linalg.norm(branch[-1] - branch[0])
    maxSampleNum = int(min(np.floor(branchLength / resolution), len(tlsPoints) / 2))
    newSkeleton = None
    newRadius = np.inf

    for sampleNum in range(maxSampleNum + 1):
        if sampleNum == 0:
            tempSkeleton = np.stack([startPoint, endPoint], axis=0)
        else: # TODO: can work on GPU if padded with np.insert + reshape instead of split (or as_strided instead of reshape)
            sections = np.array_split(order, sampleNum + 1)
            sectionsWeighted = [weightedPoints[section].sum(axis=0) for section in sections]
            sectionTotal = [densities[section].sum() for section in sections]
            samples = [(sectionsWeighted[i] + sectionsWeighted[i+1]) / (sectionTotal[i] + sectionTotal[i+1])
                       for i in range(sampleNum)]
            tempSkeleton = np.stack([startPoint] + samples + [endPoint], axis=0)

        if len(tempSkeleton) < 3:
            isCredible = True
        else:
            isCredible = np.all(getStripAngles(tempSkeleton) < smoothAngleThreshold)

        if not isCredible:
            continue

        # TODO: Idea: Each segment corresponds to a section, only calculate distance to that segment
        distances = stripPointsDistances(tempSkeleton, tlsPoints)
        tempRadius = distances.mean()
        if tempRadius < newRadius:
            newRadius = tempRadius
            newSkeleton = tempSkeleton

    if newSkeleton is not None:
        branch.skeletonPoints = newSkeleton
        branch.updateSkeletonMeasurements()
