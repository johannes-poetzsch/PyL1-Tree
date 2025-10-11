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

def adjust_bifurcations():
    eng = matlab.engine.start_matlab()
    eng.eval("load(\"L1TreeAdjustBifurcation_Input.mat\")", nargout=0)
    
    treeModel = import_tree_model(eng)
    parameters = parse_struct(eng, "parameters")
    tlsPoints = cp.array(eng.eval("tlsPoints"))  # TODO branchIDs -1
    eng.exit()

    adjustedBifurcationPoints = []
    while True:
        bifurcationID, bifurcationPoint, associatedBranches = findBifurcation(treeModel, adjustedBifurcationPoints)
        if bifurcationID is None:
            break
        
        adjustBifurcation(bifurcationID, associatedBranches, parameters)

        break


def adjustBifurcation(bifurcationID: npt.NDArray[np.double],
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
    isAtOriginalBifurcation = (cp.min(cp.stack(firstSegmentDistances.values()), axis=0) 
                               < cp.min(cp.stack(restBranchDistances.values()), axis=0))

    targetPoints = {}
    isCredible = {}
    for targetBranch in associatedBranches:
        targetPoints[targetBranch] = targetBranch.getInterpolation(parameters["resolution"]) # includes current bifurcation as invalid target point
        isCredible[targetBranch] = isBifurcationCredible(targetPoints[targetBranch], targetBranch, parentBranch, associatedBranches,
                                                         parameters["minAngleThreshold"], parameters["maxAngleThreshold"])
        
    # TODO: iterate over tlsPoints batches
    for pointBatch in [tlsPoints]:
        for targetBranch in associatedBranches:
            getDistanceMetrics(targetPoints[targetBranch], targetBranch, associatedBranches, pointBatch, isCredible[targetBranch],
                               firstSegmentDistances, restBranchDistances, isAtOriginalBifurcation)


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
    isCredible[0] = False
    return isCredible


def getDistanceMetrics(targetPoints: npt.NDArray,
                       targetBranch: Branch,
                       associatedBranches: List[Branch],
                       tlsPoints: npt.NDArray,
                       isCredible: npt.NDArray[np.bool],
                       firstSegmentDistances: Dict[Branch, npt.NDArray],
                       restBranchDistances: Dict[Branch, npt.NDArray],
                       isAtOriginalBifurcation: npt.NDArray[np.bool]):
    otherBranches = [b for b in associatedBranches if b is not targetBranch]

    # Distances to the new segments between the new bifurcation points (targetPoints) and the first remaining skeleton points of the other branches
    distancesToNewSegments = cp.min(cp.stack([linesPointsDistances(b.skeletonPoints[1], targetPoints[1:], tlsPoints) 
                                              for b in associatedBranches if b is not targetBranch]),
                                    axis=0)
    
    targetDistances = linesPointsDistances(targetPoints[:-1], # (T,N)
                                           cp.roll(targetPoints, -1, axis=0)[:-1], 
                                           tlsPoints)
    
    distancesUntilTargets = cum_min(targetDistances, axis=0, reversed=False) # (T,N)
    distancesFromTargets = cum_min(targetDistances, axis=0, reversed=True) # (T,N)
    


if __name__ == '__main__':
    adjust_bifurcations()