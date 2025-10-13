import matlab.engine
import cupy as cp
from itertools import combinations
import numpy as np
import numpy.typing as npt
import sys

from typing import Dict, List, Optional, Tuple

from src.objects.branch import Branch
from src.utils.geometry import linesPointsDistances, normalizeVectors, vecdot
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
        skip[targetBranch] = cp.count_nonzero(isCredible[targetBranch]) == 0
        
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

        relativeDistanceChange = cp.abs((meanDistancesToRemovedSegments - meanDistancesToNewBifurcation) / meanDistancesToNewBifurcation)
        relativeLengthChange = cp.abs((removedSegmentsLength[targetBranch] - newBifurcationLength[targetBranch]) / removedSegmentsLength[targetBranch])

        meetsCriteria = cp.logical_and(
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
                          maxAngleThreshold: float) -> npt.NDArray[np.bool_]:
    minAngleThreshold = np.deg2rad(minAngleThreshold)
    maxAngleThreshold = np.deg2rad(maxAngleThreshold)

    # The first target point is the current bifurcation and does not have to be checked
    # The last point does not have a next point, so it is nan
    targetPoints = targetPoints.copy()
    targetPoints[0] = cp.nan
    nextPoints = cp.roll(targetPoints, -1, axis=0)

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
            nextNextPoints = cp.roll(nextPoints, -1, axis=0)            
            
            childDirectionAtBifurcation = nextPoints - targetPoints
            childDirectionAfterBifurcation = nextNextPoints - nextPoints
        else:
            childDirectionAtBifurcation = childBranch.skeletonPoints[1] - targetPoints
            childDirectionAfterBifurcation = (childBranch.skeletonPoints[2] - childBranch.skeletonPoints[1] 
                                              if len(childBranch.skeletonPoints > 2) else None)
        
        normalizeVectors(childDirectionAtBifurcation)
        childBranchDirections.append(childDirectionAtBifurcation)
        angleAtBifurcation = cp.arccos(vecdot(parentDirection, childDirectionAtBifurcation))

        isAngleAtBifurcationCredible = angleAtBifurcation <= maxAngleThreshold
        isAngleAtBifurcationCredible[cp.isnan(angleAtBifurcation)] = True
        anglesAtBifurcationCredible.append(isAngleAtBifurcationCredible)

        if childDirectionAfterBifurcation is not None:
            normalizeVectors(childDirectionAfterBifurcation)
            angleAfterBifurcation = cp.arccos(vecdot(childDirectionAtBifurcation, childDirectionAfterBifurcation))

            isAngleAfterBifurcationCredible = angleAfterBifurcation <= maxAngleThreshold
            isAngleAfterBifurcationCredible[cp.isnan(angleAfterBifurcation)] = True
            anglesAfterBifurcationCredible.append(isAngleAfterBifurcationCredible)

    for [direction_1, direction_2] in combinations(childBranchDirections, 2):
        angleBetweenChildBranches = cp.arccos(vecdot(direction_1, direction_2))
        isAngleBetweenChildBranchesCredible = angleBetweenChildBranches > minAngleThreshold
        isAngleBetweenChildBranchesCredible[cp.isnan(angleBetweenChildBranches)] = True
        anglesBetweenChildBranchesCredible.append(isAngleBetweenChildBranchesCredible)

    isCredible = cp.all(cp.stack(anglesAtBifurcationCredible + 
                                 anglesAfterBifurcationCredible + 
                                 anglesBetweenChildBranchesCredible), axis=0)
    return isCredible[1:]


def getLengthMetrics(targetPoints: npt.NDArray,
                     otherBranches: List[Branch],
                     isCredible: npt.NDArray[np.bool]
                     ) -> Tuple[npt.NDArray, npt.NDArray]:
    newBifurcationLength = cp.sum(cp.stack([cp.linalg.norm(targetPoints[1:][isCredible] - b.skeletonPoints[1], axis=1) for b in otherBranches]), axis=0)

    targetBifurcationSegmentLength = cp.cumsum(np.linalg.norm(targetPoints - cp.roll(targetPoints, -1, axis=0), axis=1))[:-1][isCredible]
    otherBifurcationSegmentsLength = cp.sum(cp.stack([cp.linalg.norm(b.skeletonPoints[0] - b.skeletonPoints[1]) for b in otherBranches]))
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
    distancesToNewBifurcation = cp.min(cp.stack([linesPointsDistances(b.skeletonPoints[1], targetPoints[1:][isCredible], tlsPoints) 
                                                 for b in otherBranches]),
                                       axis=0)
    
    targetDistances = linesPointsDistances(targetPoints[:-1], # (T,N)
                                           cp.roll(targetPoints, -1, axis=0)[:-1], 
                                           tlsPoints)
    
    distancesUntilTargets = cum_min(targetDistances, axis=0, reversed=False) # (T,N)
    distancesToRemovedSegments = cp.minimum(distancesUntilTargets[isCredible], 
                                            cp.min(cp.stack([firstSegmentDistances[b] for b in otherBranches]), axis=0))
    del(distancesUntilTargets)

    distancesFromTargets = cum_min(targetDistances, axis=0, reversed=True) # (T,N)
    del(targetDistances)
    distancesFromTargets = cp.roll(distancesFromTargets, -1, axis=0)
    distancesFromTargets[-1] = cp.inf

    distancesToUnaffectedSegments = cp.minimum(distancesFromTargets[isCredible],
                                               cp.min(cp.stack([restBranchDistances[b] for b in otherBranches]), axis=0))
    del(distancesFromTargets)

    isAtBifurcation = cp.logical_or(distancesToRemovedSegments < distancesToUnaffectedSegments,
                                    distancesToNewBifurcation < distancesToUnaffectedSegments)

    distancesToRemovedSegmentsSum = cp.sum(cp.where(isAtBifurcation, distancesToRemovedSegments, 0), axis=1)
    distancesToNewBifurcationSum = cp.sum(cp.where(isAtBifurcation, distancesToNewBifurcation, 0), axis=1)
    distancesCount = cp.count_nonzero(isAtBifurcation, axis=1)

    newDistancesSum = cp.sum(cp.minimum(distancesToNewBifurcation, distancesToUnaffectedSegments), axis=1)

    return newDistancesSum, distancesToRemovedSegmentsSum, distancesToNewBifurcationSum, distancesCount


if __name__ == '__main__':
    adjustBifurcations()