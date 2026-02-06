import cupy as cp
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List, Optional

from .geometry import angle, normalizeVectors
from ..objects.branch import Branch


def buildSubtreeFromGraph(skeletonGraph: nx.DiGraph,
                          skeletonPoints: npt.NDArray[np.double],
                          rootIdx: int,
                          nextIdx: Optional[int] = None,
                          parent: Optional[Branch] = None) -> Branch:
    if nextIdx is None:
        if len(skeletonGraph.successors(rootIdx)) != 1:
            raise ValueError("Unclear next index for graph traversal")
        else:
            nextIdx = skeletonGraph.successors(rootIdx)[0]
    pointIndices = [rootIdx, nextIdx]

    while True:
        successors = list(skeletonGraph.successors(pointIndices[-1]))
        if len(successors) == 1:
            pointIndices.append(successors[0])
        else:
            break

    branch = Branch(id=-1,
                    skeletonIDs=np.array(pointIndices),
                    skeletonPoints=skeletonPoints[pointIndices],
                    parent=parent)

    for s in successors:
        buildSubtreeFromGraph(skeletonGraph, skeletonPoints, pointIndices[-1], s, branch)

    return branch


def condenseSkeletonPointIds(branches: List[Branch]) -> np.intp:
    maxId = 0
    for branch in branches:
        maxId = max(maxId, np.max(branch.skeletonIDs))

    idsAreUsed = np.zeros(maxId, dtype=np.intp)
    for branch in branches:
        idsAreUsed[branch.skeletonIDs] = 1

    idTable = np.cumsum(idsAreUsed) - 1
    skeletonPointCount = idTable[-1] + 1

    for branch in branches:
        branch.skeletonIDs = idTable[branch.skeletonIDs]

    return skeletonPointCount


def generateSearchRanges(branches: List[Branch], # have to be semi-ordered by distance from stem
                         useMedianForStem = False,
                         useTrimming = True) -> List[float]:
    searchRanges = []
    searchRangeLookUp = {}
    for branch in branches:
        if branch.isRemoved:
            searchRanges.append(0)
            continue

        xp = cp.get_array_module(branch.tlsPoints)
        distances = branch.getDistances(branch.tlsPoints,
                                        trimStart=useTrimming and branch.isStem(),
                                        trimEnd=useTrimming and branch.isLeaf(),
                                        returnTrimming=False)

        if useMedianForStem and branch.isStem:
            searchRange = 2 * xp.median(distances)
        else:
            searchRange = 2 * xp.quantile(distances, 0.9, method="hazen")

        if not branch.isStem():
            validParent = branch.parent
            while validParent.length <= 0.2 and not validParent.isStem(): # TODO: check magic number
                validParent = validParent.parent
            searchRange = xp.minimum(searchRange, searchRangeLookUp[validParent])

        searchRanges.append(searchRange)
        searchRangeLookUp[branch] = searchRange


def findCloseBranches(points: npt.NDArray[np.double],
                      branches: List[Branch]) -> npt.NDArray[int]:
    skeletonPoints = np.concatenate([b.skeletonPoints for b in branches], axis=0)
    branchIds = np.arange(len(branches), dtype=int).repeat([len(b.skeletonPoints) for b in branches])

    lookUp = KDTree(skeletonPoints)

    _, closestIndices = lookUp.query(points, k=1)
    closeBranchIndices = branchIds[closestIndices]

    return closeBranchIndices


def reassignAllPoints(branches: List[Branch],
                      unassignedTlsPoints: npt.NDArray[np.double],
                      useTrimming: bool = True
                      ) -> npt.NDArray[np.double]:
    prunedTlsPoints = []

    for branch in branches:
        tlsPoints = branch.tlsPoints
        branch.tlsPoints = np.empty([0,3], dtype=np.double)
        prunedTlsPoints.append(Branch.assignLocalPoints(tlsPoints, branches, closeBranches=[branch],
                                                        useTrimming=useTrimming))

    prunedTlsPoints.append(Branch.assignAllPoints(unassignedTlsPoints, branches, useTrimming=True))

    return np.concatenate(prunedTlsPoints, axis=0)


def splitAndGetEndPointIds(branch: Branch) -> List[np.intp]:
    endPoints = [branch.skeletonIDs[0], branch.skeletonIDs[-1]]
    if len(branch.skeletonIDs) < 4:
        return endPoints

    startDirection = normalizeVectors(branch.skeletonPoints[1] - branch.skeletonPoints[0])
    endDirection = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[-2])

    if angle(startDirection, endDirection) < np.pi / 2: # acute angle are fine
        return endPoints

    firstPartDirections = normalizeVectors(branch.skeletonPoints[1:-2] - branch.skeletonPoints[0])
    secondPartDirections = normalizeVectors(branch.skeletonPoints[-1] - branch.skeletonPoints[2:-1])

    splitAngles = angle(firstPartDirections, secondPartDirections)
    splitIdx = np.argmax(splitAngles) + 1

    endPoints += [branch.skeletonIDs[splitIdx], branch.skeletonIDs[splitIdx+1]]

    return endPoints
