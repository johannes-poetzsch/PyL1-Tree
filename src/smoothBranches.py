import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from typing import Dict

from objects.branch import Branch
from src.filterBranches import limitToRange
from utils.branchUtils import generateSearchRanges, reassignPoints
from utils.geometry import getStripAngles, normalizeVectors, stripPointsDistances
from utils.iteration import breadthFirstSearch
from utils.pointMetrics import calculateDensity


def smoothBranches(stem: Branch,
                   tlsPoints: npt.NDArray[np.double],
                   unassignedTlsPoints: npt.NDArray[np.double],
                   parameters: Dict[str, any]) -> npt.NDArray[np.double]:
    branches = list(breadthFirstSearch(stem))

    removeBranchAnomalies(stem)
    unassignedTlsPoints = reassignPoints(list(breadthFirstSearch(stem)), unassignedTlsPoints)
    searchRanges = generateSearchRanges(list(breadthFirstSearch(stem)), useMedianForStem=False, useTrimming=True)
    unassignedTlsPoints = np.append(unassignedTlsPoints, limitToRange(branches, searchRanges), axis=0)

    tlsKDTree = KDTree(tlsPoints)
    for branch in branches:
        densities = calculateDensity(branch.tlsPoints, tlsKDTree, parameters["initialSearchRange"]) # TODO check this
        smoothBranch(branch, densities, parameters["resolution"], np.deg2rad(parameters["smoothAngleThreshold"]))

    return unassignedTlsPoints


def removeBranchAnomalies(stem):
    rightAngle = np.deg2rad(90)
    for branch in breadthFirstSearch(stem):
        wasChanged = False
        while True:
            angles = branch.getAngles()
            if angles is None or len(angles) <= 1:
                break

            if branch.isLeaf():
                maxIdx = angles.argmax()
            else:
                maxIdx = angles[:-1].argmax() # Avoid removing the terminal point of an inner node if possible

            maxAngle = angles[maxIdx]
            if maxAngle >= rightAngle:
                deletionIdx = maxIdx + 2
                # TODO: this may be slow -> use mask instead?
                branch.skeletonPoints = np.delete(branch.skeletonPoints, deletionIdx, axis=0)
                branch.endPointIndices = None
                wasChanged = True
            else:
                break
        if angles is not None and angles[-1] > rightAngle and not branch.isLeaf():
            raise ValueError("Did not expect a terminal point to be anomalous.")
        if wasChanged:
            branch.updateSkeletonMeasurements()


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

    direction = normalizeVectors(endPoint - startPoint)

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
        branch.skeletonPoints = newSkeleton
        branch.updateSkeletonMeasurements()
