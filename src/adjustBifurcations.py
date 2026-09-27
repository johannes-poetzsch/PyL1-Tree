__all__ = ['adjustBifurcations']

import cupy as cp
import numpy as np
import numpy.typing as npt

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from src.objects.branch import Branch
from src.utils.geometry import linesPointsDistances, normalizeVectors, angle, stripPointsDistances
from src.utils.iteration import breadthFirstSearch


def adjustBifurcations(trunk: Branch, parameters: Dict[str, Any]):
    xp = np
    # using list ensures this is the original order of branches before any adjustments
    for parent in list(breadthFirstSearch(trunk, returnLeaves=False)):
        adjustmentCompleted = False
        while not adjustmentCompleted:
            if parent is None or parent.isRemoved or len(parent.children) < 2:
                break
            associatedBranches = [parent] + list(parent.children)

            # make sure bifurcation is start of each associated branch
            parent.flip()
            
            tlsPoints = xp.asarray(xp.concatenate([b.tlsPoints for b in associatedBranches]))
            adjustmentIdx, adjustmentPoint, adjustmentBranch = determineAdjustment(
                associatedBranches, parent, tlsPoints, parameters["resolution"],
                np.deg2rad(parameters["minAngleThreshold"]), np.deg2rad(parameters["maxAngleThreshold"]))

            parentOfNewBifurcation = None if adjustmentIdx is None else applyAdjustment(
                trunk, associatedBranches, adjustmentIdx, adjustmentPoint, adjustmentBranch)

            parent.unflip()

            if parentOfNewBifurcation is None:
                adjustmentCompleted = True
            else:
                # the bifurcation was merged with the previous bifurcation and the result has to be adjusted too
                parent = parentOfNewBifurcation


def applyAdjustment(stem: Branch,
                    associatedBranches: List[Branch],
                    adjustmentIdx: int,
                    adjustmentPoint: npt.NDArray[np.double],
                    adjustmentBranch: Branch
                    ) -> Optional[Branch]:
    originalDistances = [b.getDistances(b.tlsPoints) for b in associatedBranches]
    adjustmentBranch.skeletonPoints = adjustmentBranch.skeletonPoints[adjustmentIdx:]

    for branch in associatedBranches:
        branch.skeletonPoints[0] = adjustmentPoint
        branch.updateSkeletonMeasurements()
        branch.unflip()

    parentOfNewBifurcation = None
    unassignedPoints = None
    if len(adjustmentBranch) <= 1:
        # if the current parent is removed, save the new parent to adjust the new combined bifurcation
        parentOfNewBifurcation = adjustmentBranch.parent
        # branch can be removed
        adjustmentBranchIdx = associatedBranches.index(adjustmentBranch)
        del(associatedBranches[adjustmentBranchIdx])
        del(originalDistances[adjustmentBranchIdx])
        unassignedPoints = adjustmentBranch.remove()

    branches = list(breadthFirstSearch(stem))
    Branch.reassignPoints(branches, associatedBranches, originalDistances=originalDistances, useTrimming=False)

    if unassignedPoints is not None and len(unassignedPoints) > 0:
        Branch.assignLocalPoints(unassignedPoints, branches, closeBranches=associatedBranches, useTrimming=False)

    return parentOfNewBifurcation


def determineAdjustment(associatedBranches: List[Branch],
                        parentBranch: Branch,
                        tlsPoints: npt.NDArray[cp.double],
                        resolution: float,
                        minAngleThreshold: float,
                        maxAngleThreshold: float
                        ) -> Tuple[Optional[int], Optional[npt.NDArray[np.double]], Optional[Branch]]:
    targetPoints, interpolationIndices, newBifurcationLength, removedSegmentsLength, skip = {}, {}, {}, {}, {}
    isCredible: Dict[Branch, npt.NDArray[np.bool]] = {}
    for targetBranch in associatedBranches:
        otherBranches = [b for b in associatedBranches if b is not targetBranch]
        targetPoints[targetBranch], interpolationIndices[targetBranch] = targetBranch.getInterpolation(resolution) # includes current bifurcation as invalid target point
        isCredible[targetBranch] = isBifurcationCredible(targetPoints[targetBranch], targetBranch, parentBranch,
                                                         associatedBranches, minAngleThreshold, maxAngleThreshold)
        newBifurcationLength[targetBranch], removedSegmentsLength[targetBranch] = getLengthMetrics(targetPoints[targetBranch], 
                                                                                                   otherBranches, 
                                                                                                   isCredible[targetBranch])
        skip[targetBranch] = np.count_nonzero(isCredible[targetBranch]) == 0

    newMeanDistances, meanDistancesToRemovedSegments, meanDistancesToNewBifurcations = (
        getDistanceMetrics(tlsPoints, associatedBranches, targetPoints,
                           {b: np.searchsorted(interpolationIndices[b], 1) for b in associatedBranches}, isCredible))

    adjustmentIdx = adjustmentPoint = adjustmentBranch = adjustmentDistanceSum = None
    for targetBranch in associatedBranches:
        if skip[targetBranch]:
            continue

        relativeDistanceChange = np.abs((meanDistancesToRemovedSegments[targetBranch] - meanDistancesToNewBifurcations[targetBranch])
                                        / meanDistancesToNewBifurcations[targetBranch])
        relativeLengthChange = np.abs((removedSegmentsLength[targetBranch] - newBifurcationLength[targetBranch])
                                      / removedSegmentsLength[targetBranch])

        meetsCriteria = np.logical_and(
            meanDistancesToNewBifurcations[targetBranch] < meanDistancesToRemovedSegments[targetBranch],
            relativeDistanceChange > relativeLengthChange
        )

        if np.count_nonzero(meetsCriteria) == 0:
            continue

        best = np.argmin(newMeanDistances[targetBranch][meetsCriteria])
        candidateId = interpolationIndices[targetBranch][1:][isCredible[targetBranch]][meetsCriteria][best]
        candidatePoint = targetPoints[targetBranch][1:][isCredible[targetBranch]][meetsCriteria][best]
        candidateDistanceSum = newMeanDistances[targetBranch][meetsCriteria][best]

        if adjustmentIdx is None or candidateDistanceSum < adjustmentDistanceSum:
            adjustmentIdx = candidateId
            adjustmentPoint = candidatePoint
            adjustmentDistanceSum = candidateDistanceSum
            adjustmentBranch = targetBranch

    return adjustmentIdx, adjustmentPoint, adjustmentBranch


def getDistanceMetrics(tlsPoints: npt.NDArray[np.double],
                       branches: List[Branch],
                       branchSkeletonPoints: Dict[Branch, npt.NDArray],
                       bifurcationAdjacentIndices: Dict[Branch, np.intp],
                       isCredible: Dict[Branch, npt.NDArray],
                       maxBatchSize: int = 1024
                       ) -> Tuple[Dict[Branch, npt.NDArray[np.double]], Dict[Branch, npt.NDArray[np.double]], Dict[Branch, npt.NDArray[np.double]]]:
    xp = cp.get_array_module(tlsPoints)

    newDistancesSums = {b: xp.double(0) for b in branches}
    distancesToRemovedSegmentsSums = {b: xp.double(0) for b in branches}
    distancesToNewBifurcationSums = {b: xp.double(0) for b in branches}
    associatedTlsCounts = {b: xp.intp(0) for b in branches}

    batchCount = max(1, np.ceil(len(tlsPoints) / maxBatchSize).astype(np.intp))
    for batchIdx, batchPoints in enumerate(np.array_split(tlsPoints, batchCount)):
        originalDistances = {b: linesPointsDistances(branchSkeletonPoints[b][:-1],
                                                     branchSkeletonPoints[b][1:],
                                                     batchPoints) for b in branches}

        firstSegmentDistances = {b: xp.min(originalDistances[b][:bifurcationAdjacentIndices[b]], axis=0)
                                 for b in branches}
        restBranchDistances = {b: xp.min(originalDistances[b][bifurcationAdjacentIndices[b]:], axis=0, initial=np.inf)
                               for b in branches}

        for targetBranch in branches:
            otherBranches = [b for b in branches if b is not targetBranch]
            targetDistances = originalDistances[targetBranch]
            targetPoints = branchSkeletonPoints[targetBranch]
            tIsCredible = isCredible[targetBranch]
            if np.count_nonzero(tIsCredible) == 0:
                continue

            distancesToNewBifurcation = xp.min(xp.stack(
                [linesPointsDistances(targetPoints[1:][tIsCredible], b[1], batchPoints)
                 for b in otherBranches]),
                axis=0)

            # if xp is cp:
            #    distancesUntilTargets = cp.asarray(torch.as_tensor(targetDistances, device='cuda').cummin(dim=0).values)
            #    #distancesUntilTargets = cum_min(targetDistances, axis=0, reversed=False) # (T,N')
            # else:
            distancesUntilTargets = np.minimum.accumulate(targetDistances, axis=0)
            distancesToRemovedSegments = xp.minimum(distancesUntilTargets[tIsCredible], xp.min(
                xp.stack([firstSegmentDistances[b] for b in otherBranches]), axis=0))
            del distancesUntilTargets

            # if xp is cp:
            #    distancesFromTargets = cp.flipud(cp.asarray(torch.as_tensor(targetDistances, device='cuda').flipud().cummin(dim=0).values))
            #    #distancesFromTargets = cum_min(targetDistances, axis=0, reversed=True) # (T,N')
            # else:
            distancesFromTargets = np.flipud(np.minimum.accumulate(np.flipud(targetDistances)))
            del targetDistances
            distancesFromTargets = xp.roll(distancesFromTargets, -1, axis=0)
            distancesFromTargets[-1] = xp.inf

            distancesToUnaffectedSegments = xp.minimum(distancesFromTargets[tIsCredible], xp.minimum.reduce(
                [restBranchDistances[b] for b in otherBranches], axis=0))
            del distancesFromTargets

            isAtBifurcation = xp.logical_or(distancesToRemovedSegments < distancesToUnaffectedSegments,
                                            distancesToNewBifurcation < distancesToUnaffectedSegments)

            newDistancesSums[targetBranch] += xp.sum(xp.minimum(distancesToNewBifurcation, distancesToUnaffectedSegments), axis=1)
            distancesToRemovedSegmentsSums[targetBranch] += xp.sum(xp.where(isAtBifurcation, distancesToRemovedSegments, 0), axis=1)
            distancesToNewBifurcationSums[targetBranch] += xp.sum(xp.where(isAtBifurcation, distancesToNewBifurcation, 0), axis=1)
            associatedTlsCounts[targetBranch] += xp.count_nonzero(isAtBifurcation, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        meanNewDistances = {b: newDistancesSums[b] / len(tlsPoints) for b in branches}
        meanDistancesToRemovedSegments = {b: distancesToRemovedSegmentsSums[b] / associatedTlsCounts[b] for b in branches}
        meanDistancesToNewBifurcation = {b: distancesToNewBifurcationSums[b] / associatedTlsCounts[b] for b in branches}

    if xp is cp:
        meanNewDistances = {b: cp.asnumpy(meanNewDistances) for b in branches}
        meanDistancesToRemovedSegments = {b: cp.asnumpy(meanDistancesToRemovedSegments) for b in branches}
        meanDistancesToNewBifurcation = {b: cp.asnumpy(meanDistancesToNewBifurcation) for b in branches}

    return meanNewDistances, meanDistancesToRemovedSegments, meanDistancesToNewBifurcation


def getLengthMetrics(targetPoints: npt.NDArray[np.double],
                     otherBranches: List[Branch],
                     isCredible: npt.NDArray[np.bool]
                     ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    newBifurcationLength = np.sum(np.stack([np.linalg.norm(targetPoints[1:][isCredible] - b.skeletonPoints[1], axis=1) for b in otherBranches]), axis=0)

    targetBifurcationSegmentLength = np.cumsum(np.linalg.norm(targetPoints - np.roll(targetPoints, -1, axis=0), axis=1))[:-1][isCredible]
    otherBifurcationSegmentsLength = np.sum(np.stack([np.linalg.norm(b.skeletonPoints[0] - b.skeletonPoints[1], axis=-1) for b in otherBranches]))
    removedSegmentsLength = targetBifurcationSegmentLength + otherBifurcationSegmentsLength

    return newBifurcationLength, removedSegmentsLength


def isBifurcationCredible(targetPoints: npt.NDArray[np.double],
                          targetBranch: Branch,
                          parentBranch: Branch,
                          associatedBranches: List[Branch],
                          minAngleThreshold: float,
                          maxAngleThreshold: float) -> npt.NDArray[np.bool]:
    # The first target point is the current bifurcation and does not have to be checked
    # The last point does not have a next point, so it is nan
    targetPoints = targetPoints.copy()
    targetPoints[0] = np.nan
    nextPoints = np.roll(targetPoints, -1, axis=0)

    if targetBranch is parentBranch:
        parentDirection = normalizeVectors(targetPoints - nextPoints)
    else:
        parentDirection = normalizeVectors(targetPoints - parentBranch.skeletonPoints[1])

    anglesAtBifurcationCredible = []
    anglesAfterBifurcationCredible = []
    anglesBetweenChildBranchesCredible = []
    childBranchDirections = []

    for childBranch in associatedBranches:
        if childBranch is parentBranch:
            continue
        if childBranch is targetBranch:
            nextNextPoints = np.roll(nextPoints, -1, axis=0)

            childDirectionAtBifurcation = normalizeVectors(nextPoints - targetPoints)
            childDirectionAfterBifurcation = normalizeVectors(nextNextPoints - nextPoints)
        else:
            childDirectionAtBifurcation = normalizeVectors(childBranch.skeletonPoints[1] - targetPoints)

        if len(childBranch) > 2:
            childDirectionAfterBifurcation = normalizeVectors(
                childBranch.skeletonPoints[2] - childBranch.skeletonPoints[1])
        else:
            childDirectionAfterBifurcation = None

        childBranchDirections.append(childDirectionAtBifurcation)
        angleAtBifurcation = angle(parentDirection, childDirectionAtBifurcation)

        isAngleAtBifurcationCredible = angleAtBifurcation <= maxAngleThreshold
        isAngleAtBifurcationCredible[np.isnan(angleAtBifurcation)] = True
        anglesAtBifurcationCredible.append(isAngleAtBifurcationCredible)

        if childDirectionAfterBifurcation is not None:
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
