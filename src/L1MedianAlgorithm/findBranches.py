import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List

from ..utils.geometry import angle, normalizeVectors
from ..objects.branch import Branch


def findNewBranches(skeletonData: npt.NDArray,
                    kNeighbors: int,
                    distanceTracingThreshold: np.double,
                    angleTracingThreshold: np.double,
                    sigmaTracingThreshold: np.double,
                    branchSizeThreshold: int
                    ) -> List[Branch]:
    skeletonLookUp = KDTree(skeletonData["Coords"])
    neighborDistances, neighbors = skeletonLookUp.query(skeletonData["Coords"], kNeighbors + 1)
    neighborDistances, neighbors = neighborDistances[:, 1:], neighbors[:, 1:]

    newBranches = []
    descendingSigmaOrder = np.flip(np.argsort(skeletonData["Sigma"]))

    for startIdx in descendingSigmaOrder:
        if skeletonData["isBranch"][startIdx] or not skeletonData["isFix"][startIdx]:
            continue
        if skeletonData["Sigma"][startIdx] <= 0:
            break

        startNeighbors = neighbors[startIdx]
        # TODO: since only the first one is needed, looping might be faster
        neighborsAreValid = skeletonData["isFix"][startNeighbors] & ~skeletonData["isBranch"][startNeighbors]

        if np.count_nonzero(neighborsAreValid) == 0:
            skeletonData["isFix"][startIdx] = False
            continue

        secondIdx = startNeighbors[neighborsAreValid][0]
        searchDirection = normalizeVectors(skeletonData["Coords"][secondIdx] - skeletonData["Coords"][startIdx])
        newBranchForward = extendBranchInDirection(startIdx, searchDirection, skeletonData, neighbors,
                                                   distanceTracingThreshold, angleTracingThreshold)
        newBranchBackward = extendBranchInDirection(startIdx, -searchDirection, skeletonData, neighbors,
                                                    distanceTracingThreshold, angleTracingThreshold)
        newBranchBackward.reverse()
        newBranch = newBranchBackward[:-1] + newBranchForward

        if len(newBranch) < branchSizeThreshold:
            skeletonData["isFix"][startIdx] = False
            continue

        newBranch = extendBranchFromTail(newBranch, skeletonData, neighbors,
                                         distanceTracingThreshold, angleTracingThreshold, sigmaTracingThreshold)
        newBranch.reverse()
        newBranch = extendBranchFromTail(newBranch, skeletonData, neighbors,
                                         distanceTracingThreshold, angleTracingThreshold, sigmaTracingThreshold)
        newBranch.reverse()
        newBranch = np.array(newBranch)

        skeletonData["isFix"][newBranch] = True
        skeletonData["isBranch"][newBranch] = True

        newBranches.append(Branch(None, newBranch, skeletonData["Coords"][newBranch]))

    return newBranches


def extendBranchInDirection(startIdx: np.intp,
                            searchDirection: npt.NDArray[np.double],
                            skeletonData: npt.NDArray,
                            neighbors: npt.NDArray[np.intp],
                            distanceTracingThreshold: np.double,
                            angleTracingThreshold: np.double
                            ) -> List[np.intp]:
    currentIdx = startIdx
    branch = [startIdx]

    while True:
        currentPoint = skeletonData["Coords"][currentIdx]
        nextIdx = None

        for neighbor in neighbors[currentIdx]:
            if neighbor >= len(skeletonData):
                continue
            neighborPoint = skeletonData["Coords"][neighbor]
            distance = np.linalg.norm(neighborPoint - currentPoint)
            if distance > distanceTracingThreshold:
                continue
            projectedDistance = (neighborPoint - currentPoint) @ searchDirection
            if projectedDistance >= 0:
                nextIdx = neighbor
                break

        if nextIdx is None:
            break

        nextPoint = neighborPoint
        newDirection = normalizeVectors(nextPoint - currentPoint)
        searchAngle = angle(newDirection, searchDirection)

        if searchAngle > angleTracingThreshold or ~skeletonData["isFix"][nextIdx] or skeletonData["isBranch"][nextIdx]:
            break
        else:
            branch.append(nextIdx)
            searchDirection = newDirection
            currentIdx = nextIdx

    return branch


def extendBranchFromTail(branch: List[np.intp],
                         skeletonData: npt.NDArray,
                         neighbors: npt.NDArray[np.intp],
                         distanceTracingThreshold: np.double,
                         angleTracingThreshold: np.double,
                         sigmaTracingThreshold: np.double):
    while True:
        endIdx = branch[-1]
        endPoint = skeletonData["Coords"][branch[-1]]
        prevPoint = skeletonData["Coords"][branch[-2]]
        terminalDirection = normalizeVectors(endPoint - prevPoint)
        nextIdx = None

        for neighbor in neighbors[endIdx]:
            if neighbor >= len(skeletonData):
                continue
            neighborPoint = skeletonData["Coords"][neighbor]
            distance = np.linalg.norm(neighborPoint - endPoint)
            if distance > distanceTracingThreshold:
                continue
            projectedDistance = (neighborPoint - endPoint) @ terminalDirection
            if projectedDistance >= 0:
                nextIdx = neighbor
                break

        if (nextIdx is None
                or skeletonData["isBranch"][nextIdx]
                or skeletonData["Sigma"][nextIdx] < sigmaTracingThreshold):
            break

        nextPoint = skeletonData["Coords"][nextIdx]
        prevPrevPoint = skeletonData["Coords"][branch[-3]]

        prevDirection = normalizeVectors(prevPoint - prevPrevPoint)
        nextDirection = normalizeVectors(nextPoint - endPoint)

        prevAngle = angle(prevDirection, terminalDirection)
        endAngle = angle(terminalDirection, nextDirection)

        if prevAngle > angleTracingThreshold or endAngle > angleTracingThreshold:
            break

        skeletonData["isFix"][endIdx] = True
        skeletonData["isBranch"][endIdx] = True
        skeletonData["isFix"][nextIdx] = False
        skeletonData["isBranch"][nextIdx] = False

        branch.append(nextIdx)

    return branch
