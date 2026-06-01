import numpy as np
from typing import Dict, Tuple

from src.objects.branch import Branch
from src.utils.geometry import angle, normalizeVectors
from src.utils.iteration import breadthFirstSearch


def constructTreeModel(trunk: Branch) -> Tuple[Dict[Branch, float], Dict[Branch, float], Dict[Branch, bool]]:
    branchRadiiCentrifugal, radiusIsCredible = calculateBranchRadiiCentrifugal(trunk)
    continuations = getContinuations(trunk, branchRadiiCentrifugal)
    branchRadiiGaaliche = {}
    for branch in breadthFirstSearch(trunk):
        if branch.isStem() or branch is not continuations[branch.parent]:
            branchRadiiGaaliche[branch] = branchRadiiCentrifugal[branch]
        else:
            branchRadiiGaaliche[branch] = branchRadiiCentrifugal[branch.parent]
    return branchRadiiCentrifugal, branchRadiiGaaliche, radiusIsCredible


def calculateBranchRadiiCentrifugal(trunk: Branch) -> Tuple[Dict[Branch, float], Dict[Branch, bool]]:
    branchRadii: Dict[Branch, float] = {}
    radiusIsCredible: Dict[Branch, bool] = {}

    for branch in breadthFirstSearch(trunk):
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
        distances = distances[2 * parentRadius <= neighborDistances]

        branchRadii[branch] = distances.mean() if len(distances) > 0 else np.nan

    for branch in breadthFirstSearch(trunk):
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


def getContinuations(trunk: Branch, branchRadii: Dict[Branch, float]) -> Dict[Branch, Branch]:
    continuations: Dict[Branch, Branch] = {}

    for branch in breadthFirstSearch(trunk):
        if branch.isLeaf():
            continuations[branch] = branch
            continue
        branchDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[0])

        children = list(branch.children)
        childAngles = {}
        for child in branch.children:
            childDirection = normalizeVectors(child.skeletonPoints[-1] - child.skeletonPoints[0])
            childAngles[child] = np.rad2deg(angle(childDirection, branchDirection))

        continuationBranch = None
        if any([np.isnan(branchRadii[c]) for c in children]):
            minAngle = min(childAngles.values())

            children.sort(key=lambda c: branchRadii[c], reverse=True)  # descending radius
            for child in children:
                if childAngles[child] / minAngle <= 1.1:
                    continuationBranch = child
                    break
        else:
            continuationBranch = min(children, key=lambda c: childAngles[c])

        continuations[branch] = continuationBranch

    return continuations
