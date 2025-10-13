import matlab.engine
import cupy as cp
from itertools import combinations
import numpy as np
import numpy.typing as npt
import sys

from typing import Dict, List, Optional, Tuple

from src.objects.branch import Branch
from src.utils.geometry import linesPointsDistances, normalizeVectors, vecdot, angle
from src.utils.parseMatlab import parse_struct, import_tree_model
from src.utils.kernels import cum_min

def adjustBifurcations():
    eng = matlab.engine.start_matlab()
    eng.eval("load(\"L1TreeAdjustBifurcation_Input.mat\")", nargout=0)
    
    treeModel = import_tree_model(eng)
    parameters = parse_struct(eng, "parameters")
    #tlsPoints = cp.array(eng.eval("tlsPoints"))  # TODO branchIDs -1
    eng.exit()
    print("finished import")

    adjustedBifurcationPoints = []
    while True:
        bifurcationID, bifurcationPoint, associatedBranches = findBifurcation(treeModel, adjustedBifurcationPoints)
        if bifurcationID is None:
            break
        
        adjustment = determineAdjustment(bifurcationID, associatedBranches, parameters)

        break


def determineAdjustment(bifurcationID: npt.NDArray[np.double],
                        associatedBranches: List[Branch],
                        parameters: Dict[str, any]):
    parentBranch: Branch = min(associatedBranches, key=lambda b: b.cenOrder)

    for branch in associatedBranches:
        # make sure bifurcation is start of each associated branch
        branch.orientToBifurcation(bifurcationID)
        branch.toGPU()

    # TODO: divide tlsPoints into batches
    tlsPoints = cp.concatenate([b.tlsPoints for b in associatedBranches])

    firstSegmentDistances = {}
    restBranchDistances = {}
    for branch in associatedBranches:
        firstSegmentDistance, restBranchDistance = branch.getPartDistances(tlsPoints)
        firstSegmentDistances[branch] = firstSegmentDistance # (N,3)
        restBranchDistances[branch] = restBranchDistance # (N,3)

    targetPoints = {}
    newBifurcationLength = {}
    removedSegmentsLength = {}
    skip = {}
    isCredible: Dict[Branch, npt.NDArray[np.bool]] = {}
    for targetBranch in associatedBranches:
        otherBranches = [b for b in associatedBranches if b is not targetBranch]
        targetPoints[targetBranch] = targetBranch.getInterpolation(parameters["resolution"]) # includes current bifurcation as invalid target point
        isCredible[targetBranch] = isBifurcationCredible(targetPoints[targetBranch], targetBranch, parentBranch, associatedBranches,
                                                         parameters["minAngleThreshold"], parameters["maxAngleThreshold"])
        newBifurcationLength[targetBranch], removedSegmentsLength[targetBranch] = getLengthMetrics(targetPoints[targetBranch], 
                                                                                                   otherBranches, 
                                                                                                   isCredible[targetBranch])
        skip[targetBranch] = np.count_nonzero(isCredible[targetBranch]) == 0
        
    newDistancesSums = {}
    distancesToRemovedSegmentsSums = {}
    distancesToNewBifurcationSums = {}
    distancesCounts = {}
    # TODO: iterate over tlsPoints batches
    for pointBatch in [tlsPoints]:
        for targetBranch in associatedBranches:
            if skip[targetBranch]:
                continue

            otherBranches = [b for b in associatedBranches if b is not targetBranch]
            newDistancesSum, distancesToRemovedSegmentsSum, distancesToNewBifurcationSum, distancesCount = getDistanceMetrics(
                targetPoints[targetBranch], otherBranches, pointBatch, 
                isCredible[targetBranch], firstSegmentDistances, restBranchDistances)
            
            if targetBranch in distancesCounts:
                newDistancesSums[targetBranch] += newDistancesSum
                distancesToRemovedSegmentsSums[targetBranch] += distancesToRemovedSegmentsSum
                distancesToNewBifurcationSums[targetBranch] += distancesToNewBifurcationSum
                distancesCounts[targetBranch] += distancesCount
            else:
                newDistancesSums[targetBranch] = newDistancesSum
                distancesToRemovedSegmentsSums[targetBranch] = distancesToRemovedSegmentsSum
                distancesToNewBifurcationSums[targetBranch] = distancesToNewBifurcationSum
                distancesCounts[targetBranch] = distancesCount

    for targetBranch in associatedBranches:
        if skip[targetBranch]:
            continue

        meanDistancesToRemovedSegments = distancesToRemovedSegmentsSums[targetBranch] / distancesCounts[targetBranch]
        meanDistancesToNewBifurcation = distancesToNewBifurcationSums[targetBranch] / distancesCounts[targetBranch]

        relativeDistanceChange = np.abs((meanDistancesToRemovedSegments - meanDistancesToNewBifurcation) / meanDistancesToNewBifurcation)
        relativeLengthChange = np.abs((removedSegmentsLength[targetBranch] - newBifurcationLength[targetBranch]) / removedSegmentsLength[targetBranch])

        meetsCriteria = np.logical_and(
            meanDistancesToNewBifurcation < meanDistancesToRemovedSegments,
            relativeDistanceChange > relativeLengthChange
        )


def findBifurcation(treeModel: List[Branch], 
                    adjustedBifurcationPoints: List[List[np.double]]
                    ) -> Tuple[Optional[np.intp], 
                               Optional[npt.NDArray[np.double]], 
                               Optional[List[Branch]]]:
    bifurcationID: Optional[np.intp] = None
    bifurcationPoint: Optional[npt.NDArray[np.double]] = None
    associatedBranches: Optional[List[Branch]] = None
    cenOrder = sys.maxsize
    for branch in treeModel:
        if not branch.cenOrder < cenOrder:
            continue
        for i in [0,-1]:
            candidateBifurcationPoint: npt.NDArray[np.double] = branch.skeletonPoints[i]
            if candidateBifurcationPoint.tolist() in adjustedBifurcationPoints:
                continue

            candidateBifurcationID = branch.skeletonIDs[i]
            candidateAssociatedBranches = []
            for otherBranch in treeModel:
                if (otherBranch is branch or candidateBifurcationID in otherBranch.skeletonIDs[[0,-1]]):
                    candidateAssociatedBranches.append(otherBranch)
            if len(candidateAssociatedBranches) >= 3:
                bifurcationID = candidateBifurcationID
                bifurcationPoint = candidateBifurcationPoint
                associatedBranches = candidateAssociatedBranches
                cenOrder = min([b.cenOrder for b in associatedBranches])

    return bifurcationID, bifurcationPoint, associatedBranches


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
            childDirectionAfterBifurcation = (childBranch.skeletonPoints[2] - childBranch.skeletonPoints[1] 
                                              if len(childBranch.skeletonPoints > 2) else None)
        
        normalizeVectors(childDirectionAtBifurcation)
        childBranchDirections.append(childDirectionAtBifurcation)
        angleAtBifurcation = angle(parentDirection, childDirectionAtBifurcation)

        isAngleAtBifurcationCredible = angleAtBifurcation <= maxAngleThreshold
        isAngleAtBifurcationCredible[np.isnan(angleAtBifurcation)] = True
        anglesAtBifurcationCredible.append(isAngleAtBifurcationCredible)

        if childDirectionAfterBifurcation is not None:
            normalizeVectors(childDirectionAfterBifurcation)
            test = vecdot(childDirectionAtBifurcation, childDirectionAfterBifurcation)
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


def getLengthMetrics(targetPoints: npt.NDArray,
                     otherBranches: List[Branch],
                     isCredible: npt.NDArray[np.bool]
                     ) -> Tuple[npt.NDArray, npt.NDArray]:
    newBifurcationLength = np.sum(np.stack([np.linalg.norm(targetPoints[1:][isCredible] - b.skeletonPoints[1], axis=1) for b in otherBranches]), axis=0)

    targetBifurcationSegmentLength = np.cumsum(np.linalg.norm(targetPoints - np.roll(targetPoints, -1, axis=0), axis=1))[:-1][isCredible]
    otherBifurcationSegmentsLength = np.sum(np.stack([np.linalg.norm(b.skeletonPoints[0] - b.skeletonPoints[1]) for b in otherBranches]))
    removedSegmentsLength = targetBifurcationSegmentLength + otherBifurcationSegmentsLength

    return newBifurcationLength, removedSegmentsLength


def getDistanceMetrics(targetPoints: npt.NDArray,
                       otherBranches: List[Branch],
                       tlsPoints: npt.NDArray,
                       isCredible: npt.NDArray[np.bool],
                       firstSegmentDistances: Dict[Branch, npt.NDArray],
                       restBranchDistances: Dict[Branch, npt.NDArray]
                       ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    # Distances to the new segments between the new bifurcation points (targetPoints) and the first remaining skeleton points of the other branches
    xp = cp.get_array_module(tlsPoints)
    targetPoints = xp.asarray(targetPoints)
    distancesToNewBifurcation = xp.min(xp.stack([linesPointsDistances(xp.asarray(b.skeletonPoints[1]), 
                                                                      targetPoints[1:][isCredible], 
                                                                      tlsPoints)
                                                 for b in otherBranches]),
                                       axis=0)
    
    targetDistances = linesPointsDistances(targetPoints[:-1], # (T,N)
                                           xp.roll(targetPoints, -1, axis=0)[:-1], 
                                           tlsPoints)
    
    if xp is cp:
        distancesUntilTargets = cum_min(targetDistances, axis=0, reversed=False) # (T,N)
    else:
        distancesUntilTargets = np.minimum.accumulate(targetDistances, axis=0)
    distancesToRemovedSegments = xp.minimum(distancesUntilTargets[isCredible], 
                                            xp.min(xp.stack([firstSegmentDistances[b] for b in otherBranches]), axis=0))
    del(distancesUntilTargets)

    if xp is cp:
        distancesFromTargets = cum_min(targetDistances, axis=0, reversed=True) # (T,N)
    else:
        distancesFromTargets = np.flipud(np.minimum.accumulate(np.flipud(targetDistances)))
    del(targetDistances)
    distancesFromTargets = xp.roll(distancesFromTargets, -1, axis=0)
    distancesFromTargets[-1] = xp.inf

    distancesToUnaffectedSegments = xp.minimum(distancesFromTargets[isCredible],
                                               xp.min(xp.stack([restBranchDistances[b] for b in otherBranches]), axis=0))
    del(distancesFromTargets)

    isAtBifurcation = xp.logical_or(distancesToRemovedSegments < distancesToUnaffectedSegments,
                                    distancesToNewBifurcation < distancesToUnaffectedSegments)

    distancesToRemovedSegmentsSum = xp.sum(xp.where(isAtBifurcation, distancesToRemovedSegments, 0), axis=1)
    distancesToNewBifurcationSum = xp.sum(xp.where(isAtBifurcation, distancesToNewBifurcation, 0), axis=1)
    distancesCount = xp.count_nonzero(isAtBifurcation, axis=1)

    newDistancesSum = xp.sum(xp.minimum(distancesToNewBifurcation, distancesToUnaffectedSegments), axis=1)

    if xp is cp:
        newDistancesSum = cp.asnumpy(newDistancesSum)
        distancesToRemovedSegmentsSum = cp.asnumpy(distancesToRemovedSegmentsSum)
        distancesToNewBifurcationSum = cp.asnumpy(distancesToNewBifurcationSum)
        distancesCount = cp.asnumpy(distancesCount)

    return newDistancesSum, distancesToRemovedSegmentsSum, distancesToNewBifurcationSum, distancesCount


if __name__ == '__main__':
    adjustBifurcations()