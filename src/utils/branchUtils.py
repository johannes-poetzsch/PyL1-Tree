import cupy as cp
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List

from .geometry import angle, normalizeVectors
from ..objects.branch import Branch


def buildSubtreeFromGraph(skeletonGraph: nx.DiGraph,
                          skeletonPoints: npt.NDArray[np.double],
                          rootIdx: int,
                          nextIdx: int,
                          parent: Branch = None):
    pointIndices = [rootIdx, nextIdx]

    while True:
        successors = list(skeletonGraph.successors(pointIndices[-1]))
        if len(successors) == 1:
            pointIndices.append(successors[0])
        else:
            break

    branch = Branch(id=-1,
                    skeletonIDs=np.arange(len(pointIndices)),
                    skeletonPoints=skeletonPoints[pointIndices],
                    parent=parent)

    for s in successors:
        buildSubtreeFromGraph(skeletonGraph, skeletonPoints, pointIndices[-1], s, branch)

    return branch


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
            while not validParent.isStem() and validParent.length <= 0.2: # TODO: check magic number
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
