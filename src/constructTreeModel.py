from itertools import count

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple

from src.objects.branch import Branch
from src.utils.geometry import angle, normalizeVectors
from src.utils.iteration import breadthFirstSearch


cylinderDataType = [
    ("parent_cylinder_id", np.intp),
    ("branch_id_centrifugal", np.intp),
    ("branch_id_gaaliche", np.intp),
    ("branch_order_centrifugal", np.intp),
    ("branch_order_gaaliche", np.intp),
    ("start", np.double, 3),
    ("end", np.double, 3),
    ("radius", np.double)]


def constructTreeModel(trunk: Branch) -> npt.NDArray:
    branchRadiiCentrifugal, radiusIsCredible = calculateBranchRadiiCentrifugal(trunk)
    continuations = getContinuations(trunk, branchRadiiCentrifugal)
    branchRadiiGaaliche = {}
    for branch in breadthFirstSearch(trunk):
        if branch.isStem() or branch is not continuations[branch.parent]:
            branchRadiiGaaliche[branch] = branchRadiiCentrifugal[branch]
        else:
            branchRadiiGaaliche[branch] = branchRadiiCentrifugal[branch.parent]

    cylinderData = convertToCylinders(trunk, branchRadiiCentrifugal, continuations)

    return cylinderData


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

        if len(distances) == 0:
            branchRadii[branch] = np.nan
            continue

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
            validChildRadii = [branchRadii[c] for c in branch.children if not np.isnan(branchRadii[c])]
            if not validChildRadii:
                branchRadii[branch] = parentRadius
                radiusIsCredible[branch] = False
                continue

            maxChildRadius = max(validChildRadii)
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
            if minAngle == 0:
                continuationBranch = next((child for child in children if childAngles[child] == 0), None)
            else:
                for child in children:
                    if childAngles[child] / minAngle <= 1.1:
                        continuationBranch = child
                        break
        else:
            continuationBranch = min(children, key=lambda c: childAngles[c])

        continuations[branch] = continuationBranch

    return continuations


def convertToCylinders(trunk: Branch, branchRadii: Dict[Branch, float], continuations: Dict[Branch, Branch]
                       ) -> npt.NDArray:
    branches = list(breadthFirstSearch(trunk))
    cylinderCount = sum(len(b) - 1 for b in branches)
    cylinderData = np.empty(cylinderCount, dtype=cylinderDataType)

    offset = 0
    endCylinderIDs = {}
    branchOrderCentrifugal = {}
    branchOrderGaaliche = {}
    newBranchIdCentrifugal = count()
    newBranchIdGaaliche = count()
    branchIdGaaliche = {}

    for branch in branches:
        branchCylinderCount = len(branch) - 1
        endCylinderIDs[branch] = offset + branchCylinderCount - 1
        cylinderData["start"][offset:offset+branchCylinderCount] = branch[:-1]
        cylinderData["end"][offset:offset+branchCylinderCount] = branch[1:]
        cylinderData["radius"][offset:offset+branchCylinderCount] = branchRadii[branch]
        if branchCylinderCount > 1:
            cylinderData["parent_cylinder_id"][offset+1:offset+branchCylinderCount] = np.arange(offset, offset + branchCylinderCount - 1, dtype=np.intp)

        if branch.isStem():
            branchOrderCentrifugal[branch] = 0
            branchOrderGaaliche[branch] = 0
            branchIdGaaliche[branch] = next(newBranchIdGaaliche)
            cylinderData["parent_cylinder_id"][offset] = -1
        else:
            parent = branch.parent
            cylinderData["parent_cylinder_id"][offset] = endCylinderIDs[parent]
            branchOrderCentrifugal[branch] = branchOrderCentrifugal[parent] + 1

            if branch is continuations[parent]:
                branchIdGaaliche[branch] = branchIdGaaliche[parent]
                branchOrderGaaliche[branch] = branchOrderGaaliche[parent]
            else:
                branchIdGaaliche[branch] = next(newBranchIdGaaliche)
                branchOrderGaaliche[branch] = branchOrderGaaliche[parent] + 1

        cylinderData["branch_id_centrifugal"][offset:offset+branchCylinderCount] = next(newBranchIdCentrifugal)
        cylinderData["branch_id_gaaliche"][offset:offset+cylinderCount] = branchIdGaaliche[branch]
        cylinderData["branch_order_centrifugal"][offset:offset+cylinderCount] = branchOrderCentrifugal[branch]
        cylinderData["branch_order_gaaliche"][offset:offset+cylinderCount] = branchOrderGaaliche[branch]

        offset += branchCylinderCount

    return cylinderData
