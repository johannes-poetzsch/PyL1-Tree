import numpy as np
from typing import Dict, Tuple

from filterBranches import limitToRange
from objects.branch import Branch
from smoothBranches import removeBranchAnomalies
from utils.branchUtils import generateSearchRanges
#from src.utils.geometry import angle, normalizeVectors
from utils.iteration import breadthFirstSearch


def constructTreeModel(stem: Branch) -> Tuple[Dict[Branch, float], Dict[Branch, bool]]:
    branches = list(breadthFirstSearch(stem))

    removeBranchAnomalies(stem)
    Branch.reassignPoints(list(breadthFirstSearch(stem)))

    searchRanges = generateSearchRanges(list(breadthFirstSearch(stem)), useMedianForStem=False, useTrimming=True)
    limitToRange(branches, searchRanges)
    branchRadii, radiusIsCredible = calculateBranchRadiiCentrifugal(stem)
    return branchRadii, radiusIsCredible


def calculateBranchRadiiCentrifugal(stem: Branch) -> Tuple[Dict[Branch, float], Dict[Branch, bool]]:
    branchRadii: Dict[Branch, float] = {}
    radiusIsCredible: Dict[Branch, bool] = {}

    for branch in breadthFirstSearch(stem):
        if len(branch.tlsPoints) == 0:
            branchRadii[branch] = np.nan
            continue

        distances, trimmed = branch.getDistances(branch.tlsPoints, trimStart=True, trimEnd=True, returnTrimming=True)

        distances = distances[~trimmed]
        tlsPoints = branch.tlsPoints[~trimmed]

        if branch.isStem() or np.isnan(branchRadii[branch.parent]):
            branchRadii[branch] = distances.mean()
            continue

        neighborBranches = [branch.parent] + [c for c in branch.parent.children if c is not branch]
        parentRadius = branchRadii[branch.parent]

        _, neighborDistances = Branch.findClosestBranches(tlsPoints, neighborBranches, useTrimming=False)
        distances = distances[neighborDistances < 2 * parentRadius]

        branchRadii[branch] = distances.mean() if len(distances) > 0 else np.nan

    for branch in breadthFirstSearch(stem):
        if branch.isStem():
            radiusIsCredible[branch] = True
            continue

        branchRadius = branchRadii[branch]
        parentRadius = branchRadii[branch.parent]
        if not np.isnan(branchRadius) and branchRadius <= parentRadius:
            radiusIsCredible[branch] = True
            continue

        if branch.isLeaf():
            branchRadii[branch] = parentRadius
            radiusIsCredible[branch] = False
        else:
            maxChildRadius = max([branchRadii[c] for c in branch.children])
            if maxChildRadius > parentRadius:
                branchRadii[branch] = parentRadius
                radiusIsCredible[branch] = False
            else:
                branchRadii[branch] = (maxChildRadius + parentRadius) / 2
                radiusIsCredible[branch] = True

    return branchRadii, radiusIsCredible


#def calculateBranchRadiiGaaliche(stem: Branch, initialRadii: Dict[Branch, float]) -> Dict[Branch, float]:
#    branchRadii = {}
#    continuations = {}
#
#    for branch in breadthFirstSearch(stem):
#        if not branch.isLeaf():
#            continuations[branch] = getContinuation(branch, initialRadii)






#def getContinuation(branch: Branch, branchRadii: Dict[Branch, float]) -> Branch:
#    continuationBranch = None
#    branchDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[0])
#
#    childData = []
#    for child in branch.children:
#        siblingDirection = normalizeVectors(child.skeletonPoints[-1] - child.skeletonPoints[0])
#        siblingAngle = angle(siblingDirection, branchDirection)
#        childData.append({"branch": child,
#                          "radius": branchRadii[child],
#                          "angle": siblingAngle})
#
#    if any([np.isnan(branchRadii[c]) for c in branch.children]):
#        childData.sort(key=lambda cd: cd["radius"], reverse=True)  # descending radius
#        minAngle = min(cd["angle"] for cd in childData)
#        for cd in childData:
#            if cd["angle"] / minAngle <= 1.1:
#                continuationBranch = cd["branch"]
#                break
#    else:
#        continuationBranch = min(childData, key=lambda x: x["angle"])["branch"]
#
#    return continuationBranch
