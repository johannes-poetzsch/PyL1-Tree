import numpy as np
import numpy.typing as npt
from typing import List, Optional

from objects.branch import Branch
from utils.branchUtils import generateSearchRanges
from utils.geometry import angle, linesPointsDistances, normalizeVectors
from utils.iteration import breadthFirstSearch


def filterBranches(stem: Branch,
                   parameters: dict,
                   expandBranches = True) -> Optional[npt.NDArray[np.double]]:
    branches = list(breadthFirstSearch(stem))
    trimmedPoints1 = filterByLength(branches, parameters["branchSizeThreshold"], parameters["resolution"])

    branches = list(breadthFirstSearch(stem))
    prunedBranches = pruneByAngle(branches, np.deg2rad(parameters["maxAngleThreshold"]))

    trimmedPoints2 = filterByInfluence(branches, generateSearchRanges(branches, useMedianForStem=True))

    if expandBranches:
        for branch in breadthFirstSearch(stem):
            if len(branch.children) == 1:
                branch.expand()

    prunedPoints = np.concatenate([trimmedPoints1] + [trimmedPoints2] + [b.tlsPoints for b in prunedBranches], axis=0)

    return prunedPoints


def filterByLength(branches: List[Branch],
                   branchSizeThreshold: int,
                   resolution: np.double) -> npt.NDArray[np.double]:
    trimmedPoints = []

    for branch in branches:
        if (branch.length < resolution * (branchSizeThreshold - 1)
            and len(branch.skeletonPoints) < branchSizeThreshold):

            closeBranches = list(branch.children)
            if branch.parent is not None: 
                closeBranches.append(branch.parent)

            removedPoints = branch.remove()
            trimmedPoints.append(Branch.assignLocalPoints(removedPoints, branches, closeBranches, useTrimming=True))

    return np.concatenate(trimmedPoints, axis=0)


def filterByInfluence(branches: List[Branch],
                      searchRanges: List[float]) -> npt.NDArray[np.double]:
    assert(branches[0].isStem())

    trimmedPoints = []
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
                removedPoints = leaf.remove()
                closeBranches = [branch, leaf.parent]
                trimmedPoints.append(Branch.assignLocalPoints(removedPoints, branches, closeBranches, useTrimming=True))

    return np.concatenate(trimmedPoints, axis=0)


def pruneByAngle(branches: List[Branch],
                 maxAngleThreshold: float) -> List[Branch]:
    prunedBranches = []

    for branch in branches:
        if branch.isRemoved or branch.isLeaf():
            continue

        directions = normalizeVectors(branch.skeletonPoints[1:] - branch.skeletonPoints[:-1])

        for child in branch.children:
            if child.isRemoved:
                continue

            childDirection = normalizeVectors(child.skeletonPoints[-1] - child.skeletonPoints[0])
            _angle = np.median(angle(directions, childDirection))

            if _angle > maxAngleThreshold:
                for prunedBranch in breadthFirstSearch(child):
                    prunedBranch.remove(adjustConnectedBranches=False)
                    prunedBranches.append(prunedBranch)

    return prunedBranches


def pruneUncenteredBranches(branches: List[Branch]) -> List[Branch]:
    minDistance = 0.000001
    prunedBranches: List[Branch] = []

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
            prunedBranches.append(branch)

    return prunedBranches
