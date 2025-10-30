__all__ = ['adjustBifurcations']

import cupy as cp
import numpy as np
import numpy.typing as npt
import torch

from itertools import combinations
from typing import Dict, List, Optional, Tuple

from .objects.branch import Branch
from .utils.geometry import linesPointsDistances, normalizeVectors, angle
from .utils.iteration import breadthFirstSearch
#from src.utils.kernels import cum_min


def adjustBifurcations(stem, treeModel, parameters):
    xp = np

    for parent in breadthFirstSearch(stem, returnLeaves=False):
        while parent is not None and not parent.isRemoved:
            if len(parent.children) < 2:
                break

            bifurcationId = parent.skeletonIDs[-1]
            associatedBranches = [parent] + list(parent.children)

            # TODO: only flip parent
            for branch in associatedBranches:
                # make sure bifurcation is start of each associated branch
                branch.orientToBifurcation(bifurcationId)
            
            tlsPoints = xp.asarray(xp.concatenate([b.tlsPoints for b in associatedBranches]))
            adjustmentIdx, adjustmentPoint, adjustmentBranch = determineAdjustment(associatedBranches, parent, tlsPoints, parameters)

            if adjustmentIdx is None:
                parent = None
            else:
                parent = applyAdjustment(stem, associatedBranches, parent, adjustmentIdx, adjustmentPoint, adjustmentBranch, tlsPoints)

            for branch in associatedBranches:
                branch.unflip()


def determineAdjustment(associatedBranches: List[Branch],
                        parentBranch: Branch,
                        tlsPoints: npt.NDArray[cp.double],
                        parameters: Dict[str, any]
                        ) -> Tuple[Optional[int], Optional[npt.NDArray[np.double]], Optional[Branch]]:
    firstSegmentDistances = {}
    restBranchDistances = {}
    for branch in associatedBranches:
        firstSegmentDistance, restBranchDistance = branch.getPartDistances(tlsPoints)
        firstSegmentDistances[branch] = firstSegmentDistance # (N,3)
        restBranchDistances[branch] = restBranchDistance # (N,3)

    targetPoints = {}
    interpolationIndices = {}
    newBifurcationLength = {}
    removedSegmentsLength = {}
    skip = {}
    isCredible: Dict[Branch, npt.NDArray[np.bool]] = {}
    for targetBranch in associatedBranches:
        otherBranches = [b for b in associatedBranches if b is not targetBranch]
        targetPoints[targetBranch], interpolationIndices[targetBranch] = targetBranch.getInterpolation(parameters["resolution"]) # includes current bifurcation as invalid target point
        isCredible[targetBranch] = isBifurcationCredible(targetPoints[targetBranch], targetBranch, parentBranch, associatedBranches,
                                                         parameters["minAngleThreshold"], parameters["maxAngleThreshold"])
        newBifurcationLength[targetBranch], removedSegmentsLength[targetBranch] = getLengthMetrics(targetPoints[targetBranch], 
                                                                                                   otherBranches, 
                                                                                                   isCredible[targetBranch])
        skip[targetBranch] = np.count_nonzero(isCredible[targetBranch]) == 0
        
    newDistancesSums = {}
    associatedTlsToRemovedSegmentsSums = {}
    associatedTlsToNewBifurcationSums = {}
    associatedTlsCounts = {}
    # TODO: iterate over tlsPoints batches
    for pointBatch in [tlsPoints]:
        for targetBranch in associatedBranches:
            if skip[targetBranch]:
                continue

            otherBranches = [b for b in associatedBranches if b is not targetBranch]
            newDistancesSum, associatedTlsToRemovedSegmentsSum, associatedTlsToNewBifurcationSum, associatedTlsCount = getDistanceMetrics(
                targetPoints[targetBranch], otherBranches, pointBatch, 
                isCredible[targetBranch], firstSegmentDistances, restBranchDistances)
            
            if targetBranch in associatedTlsCounts:
                newDistancesSums[targetBranch] += newDistancesSum
                associatedTlsToRemovedSegmentsSums[targetBranch] += associatedTlsToRemovedSegmentsSum
                associatedTlsToNewBifurcationSums[targetBranch] += associatedTlsToNewBifurcationSum
                associatedTlsCounts[targetBranch] += associatedTlsCount
            else:
                newDistancesSums[targetBranch] = newDistancesSum
                associatedTlsToRemovedSegmentsSums[targetBranch] = associatedTlsToRemovedSegmentsSum
                associatedTlsToNewBifurcationSums[targetBranch] = associatedTlsToNewBifurcationSum
                associatedTlsCounts[targetBranch] = associatedTlsCount

    adjustmentIdx = None
    adjustmentPoint = None
    adjustmentBranch = None
    adjustmentDistanceSum = None
    for targetBranch in associatedBranches:
        if skip[targetBranch]:
            continue

        # TODO: handle division by 0
        meanDistancesToRemovedSegments = associatedTlsToRemovedSegmentsSums[targetBranch] / associatedTlsCounts[targetBranch]
        meanDistancesToNewBifurcation = associatedTlsToNewBifurcationSums[targetBranch] / associatedTlsCounts[targetBranch]

        relativeDistanceChange = np.abs((meanDistancesToRemovedSegments - meanDistancesToNewBifurcation) / meanDistancesToNewBifurcation)
        relativeLengthChange = np.abs((removedSegmentsLength[targetBranch] - newBifurcationLength[targetBranch]) / removedSegmentsLength[targetBranch])

        meetsCriteria = np.logical_and(
            meanDistancesToNewBifurcation < meanDistancesToRemovedSegments,
            relativeDistanceChange > relativeLengthChange
        )

        if np.count_nonzero(meetsCriteria) == 0:
            continue

        best = np.argmin(newDistancesSums[targetBranch][meetsCriteria])
        candidateId = interpolationIndices[targetBranch][1:][isCredible[targetBranch]][meetsCriteria][best]
        candidatePoint = targetPoints[targetBranch][1:][isCredible[targetBranch]][meetsCriteria][best]
        candidateDistanceSum = newDistancesSums[targetBranch][meetsCriteria][best]

        if adjustmentIdx is None or candidateDistanceSum < adjustmentDistanceSum:
            adjustmentIdx = candidateId
            adjustmentPoint = candidatePoint
            adjustmentDistanceSum = candidateDistanceSum
            adjustmentBranch = targetBranch

    return adjustmentIdx, adjustmentPoint, adjustmentBranch


def applyAdjustment(stem: Branch, 
                    associatedBranches: List[Branch],
                    parentBranch: Branch,
                    adjustmentIdx: np.intp, 
                    adjustmentPoint: npt.NDArray[np.double], 
                    adjustmentBranch: Branch,
                    tlsPoints: npt.NDArray[cp.double]) -> Branch:
    xp = cp.get_array_module(tlsPoints)

    originalDistances = xp.min(xp.stack([b.getDistances(tlsPoints) for b in associatedBranches]), axis=0)

    adjustmentBranch.skeletonIDs = adjustmentBranch.skeletonIDs[adjustmentIdx:]
    adjustmentBranch.skeletonPoints = adjustmentBranch.skeletonPoints[adjustmentIdx:]

    bifurcationId = adjustmentBranch.skeletonIDs[0]

    for branch in associatedBranches:
        branch.tlsPoints = np.empty([0,3], dtype=np.double)
        branch.skeletonIDs[0] = bifurcationId
        branch.skeletonPoints[0] = adjustmentPoint
        branch.updateSkeletonMeasurements()
        branch.unflip()

    newParent = None
    if adjustmentBranch.skeletonIDs.size <= 1:
        # if the current parent is removed, save the new parent to adjust the new combined bifurcation
        newParent = adjustmentBranch.parent
        # branch can be removed
        del(associatedBranches[associatedBranches.index(adjustmentBranch)])
        adjustmentBranch.remove(adjustChildSkeletons=False)
        
        if len(parentBranch.children) == 1:
            parentBranch.expand()

    unassignedPoints = Branch.reassignPoints(tlsPoints, 
                                             branches=list(breadthFirstSearch(stem)),
                                             closeBranches=associatedBranches,
                                             originalDistances=originalDistances)
    assert(len(unassignedPoints) == 0)

    return newParent


def isBifurcationCredible(targetPoints: npt.NDArray[np.double],
                          targetBranch: Branch, 
                          parentBranch: Branch, 
                          associatedBranches: List[Branch],
                          minAngleThreshold: float,
                          maxAngleThreshold: float) -> npt.NDArray[np.bool]:
    minAngleThreshold = np.deg2rad(minAngleThreshold)
    maxAngleThreshold = np.deg2rad(maxAngleThreshold)

    # The first target point is the current bifurcation and does not have to be checked
    # The last point does not have a next point, so it is nan
    targetPoints = targetPoints.copy()
    targetPoints[0] = np.nan
    nextPoints = np.roll(targetPoints, -1, axis=0)

    if targetBranch is parentBranch:
        parentDirection = targetPoints - nextPoints
    else:
        parentDirection = targetPoints - parentBranch.skeletonPoints[1]
    normalizeVectors(parentDirection)

    anglesAtBifurcationCredible = []
    anglesAfterBifurcationCredible = []
    anglesBetweenChildBranchesCredible = []
    childBranchDirections = []

    for childBranch in associatedBranches:
        if childBranch is parentBranch:
            continue
        if childBranch is targetBranch:
            nextNextPoints = np.roll(nextPoints, -1, axis=0)            
            
            childDirectionAtBifurcation = nextPoints - targetPoints
            childDirectionAfterBifurcation = nextNextPoints - nextPoints
        else:
            childDirectionAtBifurcation = childBranch.skeletonPoints[1] - targetPoints

        if len(childBranch.skeletonPoints) > 2:
            childDirectionAfterBifurcation = childBranch.skeletonPoints[2] - childBranch.skeletonPoints[1]
        else:
            childDirectionAfterBifurcation = None
        
        normalizeVectors(childDirectionAtBifurcation)
        childBranchDirections.append(childDirectionAtBifurcation)
        angleAtBifurcation = angle(parentDirection, childDirectionAtBifurcation)

        isAngleAtBifurcationCredible = angleAtBifurcation <= maxAngleThreshold
        isAngleAtBifurcationCredible[np.isnan(angleAtBifurcation)] = True
        anglesAtBifurcationCredible.append(isAngleAtBifurcationCredible)

        if childDirectionAfterBifurcation is not None:
            normalizeVectors(childDirectionAfterBifurcation)
            angleAfterBifurcation = angle(childDirectionAtBifurcation, childDirectionAfterBifurcation)

            isAngleAfterBifurcationCredible = angleAfterBifurcation <= maxAngleThreshold
            isAngleAfterBifurcationCredible[np.isnan(angleAfterBifurcation)] = True
            anglesAfterBifurcationCredible.append(isAngleAfterBifurcationCredible)

    for [direction_1, direction_2] in combinations(childBranchDirections, 2):
        angleBetweenChildBranches = angle(direction_1, direction_2)
        isAngleBetweenChildBranchesCredible = angleBetweenChildBranches > minAngleThreshold
        isAngleBetweenChildBranchesCredible[np.isnan(angleBetweenChildBranches)] = True
        anglesBetweenChildBranchesCredible.append(isAngleBetweenChildBranchesCredible)

    isCredible = np.all(np.stack(anglesAtBifurcationCredible + 
                                 anglesAfterBifurcationCredible + 
                                 anglesBetweenChildBranchesCredible), axis=0)
    return isCredible[1:]


def getLengthMetrics(targetPoints: npt.NDArray[np.double],
                     otherBranches: List[Branch],
                     isCredible: npt.NDArray[np.bool]
                     ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    newBifurcationLength = np.sum(np.stack([np.linalg.norm(targetPoints[1:][isCredible] - b.skeletonPoints[1], axis=1) for b in otherBranches]), axis=0)

    targetBifurcationSegmentLength = np.cumsum(np.linalg.norm(targetPoints - np.roll(targetPoints, -1, axis=0), axis=1))[:-1][isCredible]
    otherBifurcationSegmentsLength = np.sum(np.stack([np.linalg.norm(b.skeletonPoints[0] - b.skeletonPoints[1]) for b in otherBranches]))
    removedSegmentsLength = targetBifurcationSegmentLength + otherBifurcationSegmentsLength

    return newBifurcationLength, removedSegmentsLength


def getDistanceMetrics(targetPoints: npt.NDArray[np.double],
                       otherBranches: List[Branch],
                       tlsPoints: npt.NDArray[np.double],
                       isCredible: npt.NDArray[np.bool],
                       firstSegmentDistances: Dict[Branch, npt.NDArray],
                       restBranchDistances: Dict[Branch, npt.NDArray]
                       ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    # Distances to the new segments between the new bifurcation points (targetPoints) and the first remaining skeleton points of the other branches
    xp = cp.get_array_module(tlsPoints)
    targetPoints = xp.asarray(targetPoints)
    distancesToNewBifurcation = xp.min(xp.stack([linesPointsDistances(xp.asarray(b.skeletonPoints[1]), 
                                                                      targetPoints[1:][isCredible], 
                                                                      tlsPoints)
                                                 for b in otherBranches]),
                                       axis=0)
    
    targetDistances = linesPointsDistances(targetPoints[:-1], targetPoints[1:], tlsPoints) # (T,N)
    
    if xp is cp:
        distancesUntilTargets = cp.asarray(torch.as_tensor(targetDistances, device='cuda').cummin(dim=0).values)
        #distancesUntilTargets = cum_min(targetDistances, axis=0, reversed=False) # (T,N)
    else:
        distancesUntilTargets = np.minimum.accumulate(targetDistances, axis=0)
    distancesToRemovedSegments = xp.minimum(distancesUntilTargets[isCredible], 
                                            xp.min(xp.stack([firstSegmentDistances[b] for b in otherBranches]), axis=0))
    del(distancesUntilTargets)

    if xp is cp:
        distancesFromTargets = cp.flipud(cp.asarray(torch.as_tensor(targetDistances, device='cuda').flipud().cummin(dim=0).values))
        #distancesFromTargets = cum_min(targetDistances, axis=0, reversed=True) # (T,N)
    else:
        distancesFromTargets = np.flipud(np.minimum.accumulate(np.flipud(targetDistances)))
    del(targetDistances)
    distancesFromTargets = xp.roll(distancesFromTargets, -1, axis=0)
    distancesFromTargets[-1] = xp.inf

    # TODO: compare stack+min to minimum.reduce (adjust restBranchDistances shape in edge case)
    distancesToUnaffectedSegments = xp.minimum(distancesFromTargets[isCredible],
                                               xp.min(xp.stack([restBranchDistances[b] for b in otherBranches]), axis=0))
    del(distancesFromTargets)

    # TODO: consider adding maxEqualityDistance to avoid floating point issues
    isAtBifurcation = xp.logical_or(distancesToRemovedSegments < distancesToUnaffectedSegments,
                                    distancesToNewBifurcation < distancesToUnaffectedSegments)

    associatedTlsToRemovedSegmentsSum = xp.sum(xp.where(isAtBifurcation, distancesToRemovedSegments, 0), axis=1)
    associatedTlsToNewBifurcationSum = xp.sum(xp.where(isAtBifurcation, distancesToNewBifurcation, 0), axis=1)
    associatedTlsCount = xp.count_nonzero(isAtBifurcation, axis=1)

    newDistancesSum = xp.sum(xp.minimum(distancesToNewBifurcation, distancesToUnaffectedSegments), axis=1)

    if xp is cp:
        newDistancesSum = cp.asnumpy(newDistancesSum)
        associatedTlsToRemovedSegmentsSum = cp.asnumpy(associatedTlsToRemovedSegmentsSum)
        associatedTlsToNewBifurcationSum = cp.asnumpy(associatedTlsToNewBifurcationSum)
        associatedTlsCount = cp.asnumpy(associatedTlsCount)

    return newDistancesSum, associatedTlsToRemovedSegmentsSum, associatedTlsToNewBifurcationSum, associatedTlsCount
