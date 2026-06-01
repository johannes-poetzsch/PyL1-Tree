import numpy as np
import numpy.typing as npt
from typing import Dict, Iterable, List, Optional

from src.objects.branch import Branch
from src.utils.branchUtils import generateSearchRanges
from src.utils.geometry import angle, distanceAlongDirection, getValueOccurrences, movingTTest, normalizeVectors, \
    stripPointsDistances
from src.utils.iteration import breadthFirstSearch


def filterTreeBranches(stem: Branch,
                       parameters: dict,
                       pruneNewLeaves: bool = False) -> Optional[npt.NDArray[np.double]]:
    branches = list(breadthFirstSearch(stem))
    trimmedPoints1 = filterByLength(branches, parameters["branchSizeThreshold"], parameters["resolution"])

    branches = list(breadthFirstSearch(stem))
    prunedBranches = pruneByAngle(branches, np.deg2rad(parameters["maxAngleThreshold"]))

    trimmedPoints2 = filterByInfluence(branches, generateSearchRanges(branches, useMedianForStem=True))

    if pruneNewLeaves: # TODO: check this
        branches = list(breadthFirstSearch(stem))
        prunedBranches.extend(pruneLeafEnds(branches))
    else:
        for branch in breadthFirstSearch(stem):
            if len(branch.children) == 1:
                branch.expand()

    prunedPoints = np.concatenate([trimmedPoints1] + [trimmedPoints2] + [b.tlsPoints for b in prunedBranches], axis=0)

    return prunedPoints


def filterByLength(branches: Branch | List[Branch],
                   branchSizeThreshold: int,
                   resolution: np.double) -> npt.NDArray[np.double]:
    if not isinstance(branches, list):
        branches = breadthFirstSearch(branches)

    trimmedPoints = []
    minLength = resolution * (branchSizeThreshold - 1)
    remainingBranches = set(branches)
    for branch in branches:
        if branch.length < minLength and len(branch) < branchSizeThreshold:
            closeBranches = list(branch.children)
            if branch.parent is not None:
                closeBranches.append(branch.parent)

            removedPoints = branch.remove()
            remainingBranches.remove(branch)
            trimmedPoints.append(Branch.assignLocalPoints(removedPoints, list(remainingBranches),
                                                          closeBranches, useTrimming=True))

    return np.concatenate(trimmedPoints, axis=0) if len(trimmedPoints) > 0 else np.empty((0,3), dtype=np.double)


def filterByInfluence(branches: List[Branch],
                      searchRanges: Dict[Branch, float],
                      useMean: bool = False) -> npt.NDArray[np.double]:
    assert(branches[0].isStem())

    aggregate = np.mean if useMean else np.max

    leaves = [b for b in branches if b.isLeaf()]

    branches = branches.copy()
    branches.sort(key=lambda b: b.length, reverse=True) # descending lengths
    remainingBranches = set(branches)
    trimmedPoints = []

    for branch in branches:
        if branch.isRemoved:
            continue

        for leaf in leaves:
            if leaf is branch or leaf.isRemoved or branch.getObjectBBDistance(leaf) > searchRanges[branch]:
                continue

            distances = branch.getDistances(leaf.skeletonPoints, trimStart=True, trimEnd=True, returnTrimming=False)

            if len(distances) > len(leaf) / 2 and aggregate(distances) < searchRanges[branch]:
                removedPoints = leaf.remove()
                remainingBranches.remove(leaf)
                trimmedPoints.append(Branch.assignLocalPoints(removedPoints, list(remainingBranches),
                                                              [branch, leaf.parent], useTrimming=True))

    return np.concatenate(trimmedPoints, axis=0) if len(trimmedPoints) > 0 else np.empty((0,3), dtype=np.double)


def limitToRange(branches: Iterable[Branch],
                 pointRanges: float | Dict[Branch, float],
                 doubleBaseRange: bool = False,
                 baseLevelNumber: int = 3) -> npt.NDArray[np.double]:
    pointRanges = pointRanges if isinstance(pointRanges, dict) else {b: pointRanges for b in branches}
    prunedPoints = []

    isBaseBranch = True
    for branch in branches:
        if doubleBaseRange and isBaseBranch:
            isBaseBranch = branch.getCentrifugalOrder() <= baseLevelNumber

        pointRange = 2 * pointRanges[branch] if doubleBaseRange and isBaseBranch else pointRanges[branch]
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
        if not child.isLeaf() or len(branch) < 3 or len(child) < 3 or not branch.isEndPoint(-2):
            branch.expand()
            continue

        preEndDirection = normalizeVectors(branch.skeletonPoints[-2] - branch.skeletonPoints[-3])
        endDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[-2])
        startDirection = normalizeVectors(child.skeletonPoints[1] - child.skeletonPoints[0])

        angleBeforeEnd = np.rad2deg(angle(endDirection, preEndDirection))
        angleAtEnd = np.rad2deg(angle(startDirection, endDirection))

        pruneIdx = len(branch) - 2
        branch.expand()

        if not angleBeforeEnd + angleAtEnd > 150:
            continue
        else:
            _, prunedEnd = branch.split(pruneIdx)
            if prunedEnd is not None:
                prunedEnd.remove()
                prunedBranches.append(prunedEnd)

    return prunedBranches


def pruneThickeningLeaves(branches: List[Branch],
                          branchSizeThreshold: int,
                          useSimpleSegmentRadii = True
                          ) -> npt.NDArray[np.double]:
    prunedPoints = []

    for leaf in branches:
        if not leaf.isLeaf() or len(leaf) <= branchSizeThreshold:
            continue
        distances, pedalPoints, segmentIds = stripPointsDistances(leaf.skeletonPoints, leaf.tlsPoints, returnPedalInfo=True)
        segmentStartDistances = np.linalg.norm(pedalPoints - leaf[segmentIds], axis=1)
        segmentEndDistances = np.linalg.norm(pedalPoints - leaf[segmentIds+1], axis=1)
        closestPointIndices = np.where(segmentStartDistances < segmentEndDistances, segmentIds, segmentIds + 1)

        segmentRadii = np.full(len(leaf), np.nan, dtype=np.double)
        for skeletonIdx, indices in getValueOccurrences(closestPointIndices):
            if useSimpleSegmentRadii:
                pointDistances = distances[indices]
                segmentRadii[skeletonIdx] = np.mean(pointDistances)
            else:
                segmentPoints = leaf.tlsPoints[indices]
                skeletonPoint = leaf[skeletonIdx]
                if skeletonIdx == len(leaf) - 1: # last point has no next point
                    segmentDirection = normalizeVectors(leaf[-1] - leaf[-2])
                else:
                    segmentDirection = normalizeVectors(leaf[skeletonIdx+1] - leaf[skeletonIdx])

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
