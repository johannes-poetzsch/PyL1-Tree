from itertools import chain
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List

from .filterBranches import filterUncenteredBranches, filterByInfluence
from .objects.branch import Branch
from utils.branchUtils import splitAndGetEndPointIds


def constructTreeModel(tlsPoints: npt.NDArray[np.double],
                       branches: List[Branch],
                       parameters: Dict[str, Any]):
    Branch.assignAllPoints(tlsPoints, branches, useTrimming=True)
    branches = filterBranches(branches, parameters["baseRadius"])




def filterBranches(branches: List[Branch],
                   baseRadius: float) -> List[Branch]:
    filterUncenteredBranches(branches)
    branches = [b for b in branches if not b.isRemoved]
    # TODO: Step 1.2.2 does not work in original implementation -> implement anyways or keep behavior?

    branches.sort(key=lambda b: b.length, reverse=True)

    branchTlsDistances = [b.getDistances(b.tlsPoints) for b in branches]
    searchRanges = np.array([np.quantile(d, 0.9, method="hazen") for d in branchTlsDistances])
    filterByInfluence(branches, searchRanges=np.minimum(searchRanges, baseRadius))

    return [b for b in branches if not b.isRemoved]


def constructComponents(branches: List[Branch]) -> List[List[Branch]]:
    endPointIds = list(chain.from_iterable([splitAndGetEndPointIds(b) for b in branches]))
