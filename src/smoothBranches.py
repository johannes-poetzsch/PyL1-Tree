import cupy as cp
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from typing import Dict, List

from .objects.branch import Branch
from .utils.branchUtils import generateSearchRanges
from .utils.geometry import getStripAngles, normalizeVectors, stripPointsDistances
from .utils.iteration import breadthFirstSearch


def smoothBranches(stem: Branch, 
                   unassignedTlsPoints: npt.NDArray[np.double],
                   parameters: Dict[str, any]):
    branches = list(breadthFirstSearch(stem))

    removeBranchAnomalies(stem)
    unassignedTlsPoints = reassignAllPoints(stem, unassignedTlsPoints)
    searchRanges = {}
    generateSearchRanges(stem, searchRanges, useTrimming=True, useMedianForStem=False)
    unassignedTlsPoints = np.append(unassignedTlsPoints, limitToRange(branches, searchRanges), axis=0)
    
    allTlsPoints = [unassignedTlsPoints]
    for branch in branches:
        allTlsPoints.append(branch.tlsPoints)
    allTlsPoints = np.concatenate(allTlsPoints, axis=0)

    tlsKDTree = KDTree(allTlsPoints)
    for branch in branches:
        densities = computeDensity(branch.tlsPoints, tlsKDTree, allTlsPoints, 2 * parameters["initialSearchRange"])
        smoothBranch(branch, densities, parameters["resolution"], np.deg2rad(parameters["smoothAngleThreshold"]))


def removeBranchAnomalies(stem):
    rightAngle = np.deg2rad(90)
    for branch in breadthFirstSearch(stem):
        wasChanged = False
        while True:
            angles = branch.getAngles()
            if angles is None or len(angles) <= 1:
                break
            maxIdx = angles[:-1].argmax()
            maxAngle = angles[maxIdx]
            if maxAngle >= rightAngle:
                deletionIdx = maxIdx + 2
                # TODO: this may be slow -> use mask instead?
                branch.skeletonIDs = np.delete(branch.skeletonIDs, deletionIdx)
                branch.skeletonPoints = np.delete(branch.skeletonPoints, deletionIdx, axis=0)
                wasChanged = True
            else:
                break
        if angles is not None and angles[-1] > rightAngle:
            raise ValueError("Did not expect a terminal point to be anomalous.")
        if wasChanged:
            branch.updateSkeletonMeasurements()


def reassignAllPoints(stem: Branch, unassignedTlsPoints: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    branches = list(breadthFirstSearch(stem))
    prunedTlsPoints = []

    for branch in branches:
        tlsPoints = branch.tlsPoints
        branch.tlsPoints = np.empty([0,3], dtype=np.double)
        prunedTlsPoints.append(Branch.reassignPoints(tlsPoints, branches, closeBranches=[branch], useTrimming=True))

    # TODO: optimize performance by splitting points into local chunks
    prunedTlsPoints.append(Branch.reassignPoints(unassignedTlsPoints, branches, useTrimming=True))

    return np.concatenate(prunedTlsPoints, axis=0)


def limitToRange(branches: List[Branch], 
                 searchRanges: Dict[Branch, np.double]) -> npt.NDArray[np.double]:
    prunedTlsPoints = []

    for branch in branches:
        distances = branch.getDistances(branch.tlsPoints)
        outOfRange = distances > searchRanges[branch]
        prunedTlsPoints.append(branch.tlsPoints[outOfRange])
        branch.tlsPoints = branch.tlsPoints[~outOfRange]

    return np.concatenate(prunedTlsPoints, axis=0)


def computeDensity(points: npt.NDArray[np.double], 
                   kdTree: KDTree, 
                   allPoints: npt.NDArray[np.double], 
                   searchRange: float
                   ) -> npt.NDArray[np.double]:
    # TODO: compare to query_ball_tree, set workers
    neighbors = kdTree.query_ball_point(points, searchRange)

    densities = np.empty(shape=len(points), dtype=np.double)
    rangeFactor = -1 / ((searchRange/2) ** 2)
    for idx, neighborList in enumerate(neighbors):
        if len(neighborList) == 1: # neighborhood only contains point itself
            densities[idx] = 1
        else:
            distances: npt.NDArray = np.linalg.norm(allPoints[neighborList] - points[idx])
            distances = distances[distances > 0]
            weights: npt.NDArray = np.exp(np.square(distances) * rangeFactor)
            densities[idx] = 1 / (1 + weights.sum())

    return densities


def smoothBranch(branch: Branch, 
                 densities: npt.NDArray[np.double], 
                 resolution: float,
                 smoothAngleThreshold: float):
    _, trimmed = branch.getDistances(branch.tlsPoints, 
                                     trimStart=True, 
                                     trimEnd=True,
                                     returnTrimming=True)
    
    tlsPoints = branch.tlsPoints[~trimmed]
    if len(tlsPoints) < 10:
        return

    densities = densities[~trimmed]
    weightedPoints = tlsPoints * densities[:, np.newaxis]

    startPoint: npt.NDArray[np.double] = branch.skeletonPoints[0]
    endPoint: npt.NDArray[np.double] = branch.skeletonPoints[-1]

    direction = endPoint - startPoint
    normalizeVectors(direction)

    distanceToStart: npt.NDArray[np.double] = np.vecdot(tlsPoints - startPoint, direction)
    order = distanceToStart.argsort() # TODO: does actually ordering improve caching?
    
    maxSampleNum = int(min(np.floor(branch.length / resolution), len(tlsPoints) / 2))
    newSkeleton = None
    newRadius = np.inf

    for sampleNum in range(maxSampleNum + 1):
        if sampleNum == 0:
            tempSkeleton = np.stack([startPoint, endPoint], axis=0)
        else:
            sections = np.array_split(order, sampleNum + 1)
            sectionWeighted = [weightedPoints[section].sum(axis=0) for section in sections]
            sectionTotal = [densities[section].sum() for section in sections]
            samples = [(sectionWeighted[i] + sectionWeighted[i+1]) / (sectionTotal[i] + sectionTotal[i+1]) for i in range(sampleNum)]
            tempSkeleton = np.stack([startPoint] + samples + [endPoint], axis=0)

        if len(tempSkeleton) < 3:
            isCredible = True
        else:
            angles = getStripAngles(tempSkeleton)
            isCredible = angles.max() < smoothAngleThreshold

        if not isCredible:
            continue

        # TODO: Idea: Each segment corresponds to a section, only calculate distance to that segment
        distances = stripPointsDistances(tempSkeleton, tlsPoints)
        tempRadius = distances.mean()
        if tempRadius < newRadius:
            newRadius = tempRadius
            newSkeleton = tempSkeleton

    if newSkeleton is not None:
        branch.skeletonPoints = newSkeleton # TODO: remove skeletonIDs?
        branch.updateSkeletonMeasurements()
