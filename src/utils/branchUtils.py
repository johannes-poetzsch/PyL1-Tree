import cupy as cp
import numpy as np
import numpy.typing as npt
from typing import Dict, Optional

from ..objects.branch import Branch


def generateSearchRanges(branch: Branch,
                         searchRanges: Dict[Branch, np.double],
                         minAncestorSearchRange: Optional[np.double] = np.inf,
                         useTrimming = True,
                         useMedianForStem = True):
    if branch.isRemoved:
        return

    xp = cp.get_array_module(branch.tlsPoints)
    if useTrimming:
        distances = branch.getDistances(branch.tlsPoints, 
                                        trimStart=useTrimming and branch.isStem(), 
                                        trimEnd=useTrimming and branch.isLeaf(),
                                        returnTrimming=False)

    if useMedianForStem and branch.isStem:
        searchRange = 2 * xp.median(distances)
    else:
        searchRange = 2 * xp.quantile(distances, 0.9, method="midpoint") # TODO: check alternatives

    searchRange = xp.minimum(searchRange, minAncestorSearchRange)

    searchRanges[branch] = searchRange
    for child in branch.children:
        generateSearchRanges(child, searchRanges, 
                             searchRange if branch.length > 0.2 else minAncestorSearchRange, # TODO: check magic number
                             useTrimming=useTrimming)
