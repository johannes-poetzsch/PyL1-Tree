import numpy as np
import numpy.typing as npt
from itertools import repeat
from typing import Iterable, List, Optional

from src.objects.branch import Branch
from src.utils.branchUtils import generateSearchRanges
from src.utils.geometry import angle, distanceAlongDirection, movingTTest, normalizeVectors, stripPointsDistances
from src.utils.iteration import breadthFirstSearch


def filterTreeBranches(stem: Branch,
                       parameters: dict,
                       pruneNewLeaves: bool = False) -> Optional[npt.NDArray[np.double]]:
    branches = list(breadthFirstSearch(stem))
    trimmedPoints1 = filterByLength(branches, parameters["branchSizeThreshold"], parameters["resolution"])

    branches = list(breadthFirstSearch(stem))
    prunedBranches = pruneByAngle(branches, np.deg2rad(parameters["maxAngleThreshold"]))

    trimmedPoints2 = filterByInfluence(branches, generateSearchRanges(branches, useMedianForStem=True))

    if pruneNewLeaves:
        branches = list(breadthFirstSearch(stem))
        prunedBranches.extend(pruneLeafEnds(branches))
    else:
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

    return np.concatenate(trimmedPoints, axis=0) if len(trimmedPoints) > 0 else np.empty((0,3), dtype=np.double)


def filterByInfluence(branches: List[Branch],
                      searchRanges: List[float],
                      useMean: bool = False) -> npt.NDArray[np.double]:
    assert(branches[0].isStem())

    aggregate = np.mean if useMean else np.max

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

            if len(distances) > len(leaf.skeletonPoints) / 2 and aggregate(distances) < searchRange:
                removedPoints = leaf.remove()
                closeBranches = [branch, leaf.parent]
                trimmedPoints.append(Branch.assignLocalPoints(removedPoints, branches, closeBranches, useTrimming=True))

    return np.concatenate(trimmedPoints, axis=0) if len(trimmedPoints) > 0 else np.empty((0,3), dtype=np.double)


def limitToRange(branches: Iterable[Branch],
                 pointRanges: float | Iterable[float],
                 doubleBaseRange: bool = False,
                 baseLevelNumber: int = 3) -> npt.NDArray[np.double]:
    ranges = pointRanges if isinstance(pointRanges, Iterable) else repeat(pointRanges)
    prunedPoints = []

    isBaseBranch = True
    for branch, pointRange in zip(branches, ranges):
        if doubleBaseRange and isBaseBranch:
            isBaseBranch = branch.getCentrifugalOrder() <= baseLevelNumber

        pointRange = 2 * pointRange if doubleBaseRange and isBaseBranch else pointRange
        distances = branch.getDistances(branch.tlsPoints)

        isInRange = distances <= pointRange
        prunedPoints.append(branch.tlsPoints[~isInRange])
        branch.tlsPoints = branch.tlsPoints[isInRange]

    return np.concatenate(prunedPoints, axis=0) if len(prunedPoints) > 0 else np.empty((0,3), dtype=np.double)


def pruneByAngle(branches: List[Branch],
                 maxAngleThreshold: float) -> List[Branch]:
    prunedBranches = []

    for branch in branches:
        if branch.isRemoved or branch.isLeaf():
            continue

        directions = normalizeVectors(branch.skeletonPoints[1:] - branch.skeletonPoints[:-1])

        for child in list(branch.children):
            if child.isRemoved:
                continue

            childDirection = normalizeVectors(child.skeletonPoints[-1] - child.skeletonPoints[0])
            _angle = np.median(angle(directions, childDirection))

            if _angle > maxAngleThreshold:
                for prunedBranch in breadthFirstSearch(child):
                    prunedBranch.remove(adjustConnectedBranches=False)
                    prunedBranches.append(prunedBranch)

    return prunedBranches


def pruneLeafEnds(branches: List[Branch]) -> List[Branch]:
    prunedBranches = []
    for branch in branches:
        if len(branch.children) != 1:
            continue
        child = next(iter(branch.children))
        if (not child.isLeaf() or
                len(branch.skeletonPoints) < 3 or
                len(child.skeletonPoints) < 3 or
                not branch.isEndPoint(-2)):
            branch.expand()
            continue

        preEndDirection = normalizeVectors(branch.skeletonPoints[-2] - branch.skeletonPoints[-3])
        endDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[-2])
        startDirection = normalizeVectors(child.skeletonPoints[1] - child.skeletonPoints[0])

        angleBeforeEnd = np.rad2deg(angle(endDirection, preEndDirection))
        angleAtEnd = np.rad2deg(angle(startDirection, endDirection))

        pruneIdx = len(branch.skeletonPoints) - 2
        branch.expand()

        if not angleBeforeEnd + angleAtEnd > 150:
            continue
        else:
            _, prunedEnd = branch.split(pruneIdx)
            if prunedEnd is not None:
                prunedEnd.remove()
                prunedBranches.append(prunedEnd)

    return prunedBranches


def pruneThickeningLeaves(branches: List[Branch], branchSizeThreshold: int) -> npt.NDArray[np.double]:
    prunedPoints = []

    for leaf in branches:
        if not leaf.isLeaf() or len(leaf.skeletonPoints) <= branchSizeThreshold:
            continue
        _, _, closestPointIndices = stripPointsDistances(leaf.skeletonPoints, leaf.tlsPoints, returnPedalInfo=True)

        order = np.argsort(closestPointIndices)
        tlsPoints = leaf.tlsPoints[order]
        skeletonPoints = leaf.skeletonPoints
        closestPointIndices = closestPointIndices[order]
        # TODO: test and fix this
        skeletonIndices, indices, counts = np.unique(closestPointIndices, return_index=True, return_counts=True)

        segmentRadii = np.full(len(skeletonPoints), np.nan, dtype=np.double)
        for skeletonIdx, idx, count in zip(skeletonIndices, indices, counts):
            skeletonPoint = skeletonPoints[skeletonIdx]
            segmentPoints = tlsPoints[idx:idx + count]
            if skeletonIdx == len(skeletonPoints) - 1: # last point has no next point
                segmentDirection = normalizeVectors(skeletonPoints[-1] - skeletonPoints[-2])
            else:
                segmentDirection = normalizeVectors(skeletonPoints[skeletonIdx+1] - skeletonPoints[skeletonIdx])

            distances = np.linalg.norm(segmentPoints - skeletonPoint, axis=1)
            baseDistances = np.abs(distanceAlongDirection(skeletonPoint, segmentPoints, segmentDirection))
            pedalDistances = np.sqrt(distances ** 2 - baseDistances ** 2)
            segmentRadii[skeletonIdx] = np.real(np.mean(pedalDistances))

        breakIdx = movingTTest(segmentRadii)
        if breakIdx is None or breakIdx < branchSizeThreshold:
            continue

        leaf, prunedEnd = leaf.split(breakIdx)
        if prunedEnd is not None:
            prunedPoints.append(prunedEnd.remove())

    return np.concatenate(prunedPoints, axis=0) if len(prunedPoints) > 0 else np.empty((0,3), dtype=np.double)


def pruneUncenteredBranches(branches: List[Branch]) -> List[Branch]:
    minDistance = 0.000001
    prunedBranches: List[Branch] = []

    for branch in branches:
        if branch.isRemoved:
            continue
        distances, pedalPoints, _ = stripPointsDistances(branch.skeletonPoints, branch.tlsPoints, returnPedalInfo=True)

        tlsPoints = branch.tlsPoints[distances > minDistance]
        pedalPoints = pedalPoints[distances > minDistance]
        if len(tlsPoints) == 0:
            branch.remove()
            prunedBranches.append(branch)
            continue

        branchDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[0])

        coordinateTransformation = np.empty([3,3], dtype=np.double)
        coordinateTransformation[2] = branchDirection
        coordinateTransformation[0] = normalizeVectors(tlsPoints[0] - pedalPoints[0])
        coordinateTransformation[1] = normalizeVectors(np.cross(coordinateTransformation[0],
                                                                coordinateTransformation[2]))

        oldCoords = branch.skeletonPoints[0] + tlsPoints - pedalPoints
        newCoords = (oldCoords - branch.skeletonPoints[0]) @ np.linalg.inv(coordinateTransformation)
        radialDirections = normalizeVectors(newCoords[:,:2])

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
