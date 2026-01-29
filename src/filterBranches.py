import cupy as cp
import numpy as np
import numpy.typing as npt

from typing import Dict, List, Optional, Tuple

from .objects.branch import Branch
from .utils.branchUtils import generateSearchRanges
from .utils.geometry import angle, linesPointsDistances, normalizeVectors
from .utils.iteration import breadthFirstSearch


def filterBranches(stem: Branch, parameters: dict) -> Optional[npt.NDArray[np.double]]:
    branches = list(breadthFirstSearch(stem))
    removedBranchesLength, closeBranchesLength = filterByLength(branches,
                                                                parameters["branchSizeThreshold"],
                                                                parameters["resolution"])
    prunedBranches, _ = filterByAngle(branches, parameters["maxAngleThreshold"])

    branches = [b for b in branches if not b.isRemoved]
    removedBranchesInfluence, closeBranchesInfluence = (
        filterByInfluence(branches,generateSearchRanges(branches, useMedianForStem=True)))

    removedBranches = removedBranchesLength + removedBranchesInfluence
    closeBranches = closeBranchesLength + closeBranchesInfluence
    prunedTlsPoints = []

    branches = list(breadthFirstSearch(stem)) + prunedBranches
    for rb, cb in zip(removedBranches, closeBranches):
        prunedTlsPoints.append(Branch.assignLocalPoints(rb.tlsPoints, branches, cb))

    if len(prunedTlsPoints) == 0:
        return None
    else:
        return np.concatenate(prunedTlsPoints, axis=0)


def filterByLength(branches: List[Branch],
                   branchSizeThreshold: int,
                   resolution: np.double) -> Tuple[List[Branch], List[List[Branch]]]:
    removedBranches: List[Branch] = []
    closeBranches: List[List[Branch]] = []

    for branch in branches:
        if (branch.length < resolution * (branchSizeThreshold - 1) 
            and len(branch.skeletonIDs) < branchSizeThreshold):

            closeBranches.append(list(branch.children))
            if branch.parent is not None: 
                closeBranches[-1].append(branch.parent)

            branch.remove()
            removedBranches.append(branch)

    return removedBranches, closeBranches


def filterByAngle(branches: List[Branch],
                  maxAngleThreshold: float) -> Tuple[List[Branch], List[List[Branch]]]:
    maxAngleThreshold = np.deg2rad(maxAngleThreshold)
    removedBranches = []

    for branch in branches:
        if branch.isRemoved or branch.isLeaf():
            continue

        parentDirections = normalizeVectors(branch.skeletonPoints[1:] - branch.skeletonPoints[:-1])

        for child in branch.children:
            if child.isRemoved:
                continue

            childDirection = normalizeVectors(child.skeletonPoints[-1] - child.skeletonPoints[0])
            _angle = np.median(angle(parentDirections, childDirection))

            if _angle > maxAngleThreshold:
                for prunedBranch in breadthFirstSearch(child):
                    prunedBranch.remove(adjustConnectedBranches=False)
                    removedBranches.append(prunedBranch)

    return removedBranches, [[]] * len(removedBranches)


def filterByInfluence(branches: List[Branch],
                      searchRanges: List[float]) -> Tuple[List[Branch], List[List[Branch]]]:
    removedBranches: List[Branch] = []
    closeBranches: List[List[Branch]] = []

    assert(branches[0].isStem())
    leaves = [b for b in branches if b.isLeaf()]

    branches.sort(key=lambda b: b.length, reverse=True) # descending lengths

    for branch, searchRange in zip(branches, searchRanges):
        if branch.isRemoved:
            continue

        for leaf in leaves:
            if leaf is branch or leaf.isRemoved or branch.branchBoundingBoxDistance(leaf) >= searchRange:
                continue

            distances = branch.getDistances(leaf.skeletonPoints, trimStart=True, trimEnd=True, returnTrimming=False)

            if len(distances) > len(leaf.skeletonPoints) / 2 and distances.max() < searchRange:
                leaf.remove()
                removedBranches.append(leaf)
                closeBranches.append([branch, leaf.parent])
    
    return removedBranches, closeBranches


def filterUncenteredBranches(branches: List[Branch]) -> Tuple[List[Branch], List[List[Branch]]]:
    minDistance = 0.000001
    removedBranches: List[Branch] = []

    for branch in branches:
        if branch.isRemoved:
            continue
        distances, pedalPoints = linesPointsDistances(branch.skeletonPoints[:-1],
                                                      branch.skeletonPoints[1:],
                                                      branch.tlsPoints,
                                                      returnPedalPoints=True)

        tlsPoints = branch.tlsPoints[distances > minDistance]
        pedalPoints = pedalPoints[distances > minDistance]
        if len(tlsPoints) <= 1:
            continue

        branchDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[0])

        coordinateTransformation = np.empty([3,3], dtype=np.double)
        coordinateTransformation[2] = branchDirection
        coordinateTransformation[0] = tlsPoints[0] - pedalPoints[0]
        coordinateTransformation[1] = normalizeVectors(np.cross(coordinateTransformation[0],
                                                                coordinateTransformation[2]))

        oldCoords = branch.skeletonPoints[0] + tlsPoints - pedalPoints
        newCoords = (oldCoords - branch.skeletonPoints[0]) @ np.linalg.inv(coordinateTransformation)
        radialDirections = normalizeVectors(newCoords[:2])

        angles = np.rad2deg(np.acos(radialDirections[:,0]))
        isReflexAngle = radialDirections[:,1] < 0.0
        angles[isReflexAngle] = 360 - angles[isReflexAngle]

        angles.sort()

        angleDiff = angles - np.roll(angles, 1)
        angleDiff[0] += 360

        angleDiff[::-1].sort() # sort in descending order

        if angleDiff[1] * np.sqrt(len(angles)) > 720:
            branch.remove()
            removedBranches.append(branch)

    return removedBranches, [[]] * len(removedBranches)
