import cupy as cp
import numpy as np
import numpy.typing as npt

from typing import Dict, List, Optional, Tuple

from .objects.branch import Branch
from .utils.geometry import angle, normalizeVectors, stripPointsDistances
from .utils.iteration import breadthFirstSearch


def filterBranches(stem: Branch, parameters: dict) -> Optional[npt.NDArray[np.double]]:
    unassignedTlsPoints_1, closeBranches_1 = filterByLength(stem, parameters["branchSizeThreshold"], parameters["resolution"])
    pruneByAngle(stem, parameters["maxAngleThreshold"])
    unassignedTlsPoints_2, closeBranches_2 = filterByInfluence(stem)

    unassignedTlsPoints = unassignedTlsPoints_1 + unassignedTlsPoints_2
    closeBranches = closeBranches_1 + closeBranches_2
    prunedTlsPoints = []

    branches = list(breadthFirstSearch(stem))
    for p, b in zip(unassignedTlsPoints, closeBranches):
        Branch.reassignPoints(p, branches, b)

    for branch in branches:
        if branch.isRemoved:
            prunedTlsPoints.append(branch.remove())

    if len(prunedTlsPoints) == 0:
        return None
    else:
        return np.concatenate(prunedTlsPoints)


def filterByLength(stem: Branch,
                   branchSizeThreshold: int,
                   resolution: np.double) -> Tuple[List[npt.NDArray[np.double]], List[List[Branch]]]:
    unassignedTlsPoints: List[npt.NDArray[np.double]] = []
    closeBranches: List[List[Branch]] = []

    for branch in breadthFirstSearch(stem):
        if (branch.length < resolution * (branchSizeThreshold - 1) 
            and len(branch.skeletonIDs) < branchSizeThreshold):

            closeBranches.append(list(branch.children))
            if branch.parent is not None: 
                closeBranches[-1].append(branch.parent)

            unassignedTlsPoints.append(branch.remove(adjustChildSkeletons=True))

    return unassignedTlsPoints, closeBranches


def pruneByAngle(stem: Branch,
                  maxAngleThreshold: float):
    maxAngleThreshold = np.deg2rad(maxAngleThreshold)

    for branch in breadthFirstSearch(stem, returnLeaves=False):
        if branch.isRemoved:
            continue

        parentDirections = branch.skeletonPoints[1:] - branch.skeletonPoints[:-1]
        normalizeVectors(parentDirections)

        for child in branch.children:
            if child.isRemoved:
                continue

            childDirection = child.skeletonPoints[-1] - child.skeletonPoints[0]
            normalizeVectors(childDirection)
            _angle = np.median(angle(parentDirections, childDirection))

            if _angle > maxAngleThreshold:
                for prunedBranch in breadthFirstSearch(child):
                    prunedBranch.isRemoved = True


def filterByInfluence(stem: Branch) -> Tuple[List[npt.NDArray[np.double]], List[List[Branch]]]:
    unassignedTlsPoints: List[npt.NDArray[np.double]] = []
    closeBranches: List[List[Branch]] = []

    searchRanges = {}
    generateSearchRanges(stem, searchRanges)
    branches = [b for b in breadthFirstSearch(stem) if not b.isRemoved]
    leaves = [b for b in branches if b.isLeaf()]

    branches.sort(key=lambda b: b.length, reverse=True) # descending lengths

    for branch in branches:
        if branch.isRemoved:
            continue

        searchRange = searchRanges[branch]

        for leaf in leaves:
            if leaf.isRemoved or branch.branchBoundingBoxDistance(leaf) >= searchRange:
                continue

            distances = stripPointsDistances(branch.skeletonPoints, leaf.skeletonPoints, trimStart=True, trimEnd=True)

            if len(distances) > len(leaf.skeletonPoints) / 2 and distances.max() < searchRange:
                closeBranches.append([branch, leaf.parent])
                unassignedTlsPoints.append(leaf.remove())
    
    return unassignedTlsPoints, closeBranches


def generateSearchRanges(branch: Branch,
                         searchRanges: Dict[Branch, np.double],
                         minAncestorSearchRange: Optional[np.double] = None):
    if branch.isRemoved:
        return

    xp = cp.get_array_module(branch.tlsPoints)
    if minAncestorSearchRange is None: # stem
        distances: npt.NDArray = stripPointsDistances(xp.asarray(branch.skeletonPoints), branch.tlsPoints, trimStart=True)
        searchRange = 2 * xp.median(distances)
        searchRanges[branch] = searchRange
        for child in branch.children:
            generateSearchRanges(child, searchRanges, searchRange)
    else: # not the stem
        distances = npt.NDArray = stripPointsDistances(xp.asarray(branch.skeletonPoints), branch.tlsPoints, trimEnd=branch.isLeaf())
        searchRange = 2 * xp.quantile(distances, 0.9, method="midpoint") # TODO: check alternatives
        searchRange = xp.minimum(searchRange, minAncestorSearchRange)
        searchRanges[branch] = searchRange
        for child in branch.children:
            generateSearchRanges(child, searchRanges, searchRange if branch.length > 0.2 else minAncestorSearchRange) # TODO: check magic number
