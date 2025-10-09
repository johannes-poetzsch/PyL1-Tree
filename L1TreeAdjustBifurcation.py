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

    for targetBranch in associatedBranches:
        getDistanceMetrics(targetPoints[1:], targetBranch, associatedBranches)
        



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


def getDistanceMetrics(associatedBranches: List[Branch],
                       parentBranch: Branch,
                       tlsPoints: npt.NDArray,
                       parameters: Dict[str, any]):
    # TODO: check if points already belong to nearest branch
    firstSegmentDistances = {}
    restBranchDistances = {}
    #branchDistances = []
    for branch in associatedBranches:
        firstSegmentDistance, restBranchDistance = branch.getPartDistances(tlsPoints)
        firstSegmentDistances[branch] = firstSegmentDistance # (N,3)
        restBranchDistances[branch] = restBranchDistance # (N,3)
        #branchDistances.append(cp.minimum(firstSegmentDistance, restBranchDistance))

    #originalDistances = cp.min(cp.stack(branchDistances), axis=0)
    isAtOriginalBifurcation = cp.min(firstSegmentDistances.values(), axis=0) < cp.min(restBranchDistances.values(), axis=0)

    for targetBranch in associatedBranches:
        otherBranches = [b for b in associatedBranches if b is not targetBranch]
        targetPoints = targetBranch.getInterpolation(parameters["resolution"]) # includes current bifurcation as invalid target point
        isCredible = isBifurcationCredible(targetPoints, targetBranch, parentBranch, associatedBranches,
                                           parameters["minAngleThreshold"], parameters["maxAngleThreshold"])

        # Distances to the new segments between the new bifurcation points (targetPoints) and the first remaining skeleton points of the other branches
        distancesToNewSegments = cp.min(cp.stack([linesPointsDistances(b.skeletonPoints[1], targetPoints[1:], tlsPoints) 
                                                  for b in associatedBranches if b is not targetBranch]),
                                        axis=0)
        
        targetDistances = linesPointsDistances(targetPoints[:-1], 
                                               cp.roll(targetPoints, -1, axis=0)[:-1], 
                                               tlsPoints)  
    


if __name__ == '__main__':
    adjust_bifurcations()