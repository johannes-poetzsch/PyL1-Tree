import networkx as nx
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional

from src.objects.branch import Branch
from src.utils.gpuAcceleration import array_module


def buildSubtreeFromGraph(skeletonGraph: nx.DiGraph,
                          skeletonPoints: npt.NDArray[np.double],
                          startIdx: int,
                          nextIdx: Optional[int] = None,
                          parent: Optional[Branch] = None,
                          isEndPoint: Optional[npt.NDArray[bool]] = None
                          ) -> Branch:
    if nextIdx is None:
        successors = list(skeletonGraph.successors(startIdx))
        if len(successors) != 1:
            raise ValueError("Unclear next index for graph traversal")
        else:
            nextIdx = successors[0]
    pointIndices = [startIdx, nextIdx]

    while True:
        successors = list(skeletonGraph.successors(pointIndices[-1]))
        if len(successors) == 1:
            pointIndices.append(successors[0])
        else:
            break

    branch = Branch(skeletonPoints[pointIndices], parent=parent)

    pointIndices = np.array(pointIndices, dtype=int)
    if isEndPoint is not None:
        Branch.trackEndPoints = True
        endPointIndices = np.flatnonzero(isEndPoint[pointIndices])
        if len(endPointIndices) > 2:
            branch.addEndPoints(endPointIndices[1:-1])

    for s in successors:
        buildSubtreeFromGraph(skeletonGraph, skeletonPoints, startIdx=pointIndices[-1], nextIdx=s, parent=branch,
                              isEndPoint=isEndPoint)

    return branch


def generateSearchRanges(branches: List[Branch], # have to be semi-ordered by distance from stem
                         useMedianForStem = False,
                         useTrimming = True) -> Dict[Branch, float]:
    searchRanges = {}
    for branch in branches:
        if branch.isRemoved:
            searchRanges[branch] = 0.0
            continue

        xp = array_module(branch.tlsPoints)
        distances = branch.getDistances(branch.tlsPoints,
                                        trimStart=useTrimming and branch.isStem(),
                                        trimEnd=useTrimming and branch.isLeaf())

        if len(distances) == 0:
            searchRanges[branch] = np.nan
            continue

        if useMedianForStem and branch.isStem():
            searchRange = 2 * xp.median(distances)
        else:
            searchRange = 2 * xp.quantile(distances, 0.9, method="hazen")

        if not branch.isStem():
            validParent = branch.parent
            while validParent.length <= 0.2 and not validParent.isStem(): # TODO: check magic number
                validParent = validParent.parent
            if searchRanges[validParent] < searchRange:
                searchRange = searchRanges[validParent]

        searchRanges[branch] = searchRange

    return searchRanges
