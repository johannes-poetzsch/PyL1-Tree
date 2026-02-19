import cupy as cp
from itertools import repeat
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import List, Optional

from ..objects.branch import Branch


def buildSubtreeFromGraph(skeletonGraph: nx.DiGraph,
                          skeletonPoints: npt.NDArray[np.double],
                          rootIdx: int,
                          nextIdx: Optional[int] = None,
                          parent: Optional[Branch] = None,
                          isEndPoint: Optional[npt.NDArray[bool]] = None
                          ) -> Branch:
    if nextIdx is None:
        successors = list(skeletonGraph.successors(rootIdx))
        if len(successors) != 1:
            raise ValueError("Unclear next index for graph traversal")
        else:
            nextIdx = successors[0]
    pointIndices = [rootIdx, nextIdx]

    while True:
        successors = list(skeletonGraph.successors(pointIndices[-1]))
        if len(successors) == 1:
            pointIndices.append(successors[0])
        else:
            break

    branch = Branch(skeletonPoints[pointIndices], parent=parent)

    if isEndPoint is not None:
        endPointIndices = pointIndices[isEndPoint[pointIndices]]
        if len(endPointIndices) > 2:
            branch.addEndPoints(endPointIndices[1:-1])

    for s in successors:
        buildSubtreeFromGraph(skeletonGraph, skeletonPoints,
                              rootIdx=pointIndices[-1], nextIdx=s,
                              parent=branch, isEndPoint=isEndPoint)

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


def reassignPoints(branches: List[Branch],
                   unassignedTlsPoints: Optional[npt.NDArray[np.double]] = None,
                   changedBranches: Optional[List[Branch]] = None,
                   originalDistances: Optional[List[npt.NDArray[np.double]]] = None,
                   useTrimming: bool = True
                   ) -> npt.NDArray[np.double]:
    changedBranches = branches if changedBranches is None else changedBranches
    originalDistances = repeat(None) if originalDistances is None else originalDistances
    prunedTlsPoints = []

    for branch, od in zip(changedBranches, originalDistances):
        tlsPoints = branch.tlsPoints
        branch.tlsPoints = np.empty([0,3], dtype=np.double)
        prunedTlsPoints.append(Branch.assignLocalPoints(tlsPoints, branches, closeBranches=[branch],
                                                        originalDistances=od,
                                                        useTrimming=useTrimming))

    if unassignedTlsPoints is not None:
        prunedTlsPoints.append(Branch.assignAllPoints(unassignedTlsPoints, branches, useTrimming=True))

    return np.concatenate(prunedTlsPoints, axis=0)
