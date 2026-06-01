import numpy as np

from src.objects.branch import Branch
from src.utils.geometry import pointSort


def branches_are_equal(branch1: Branch, branch2: Branch, compareTlsPoints: bool = True, atol: float = 0) -> bool:
    if len(branch1) != len(branch2) or not np.allclose(branch1.skeletonPoints, branch2.skeletonPoints, atol=atol):
        return False

    if compareTlsPoints:
        if len(branch1.tlsPoints) != len(branch2.tlsPoints):
            for branch in [branch1, branch2]:
                isStartPoint = np.all(branch.tlsPoints == branch[0], axis=1)
                isEndPoint = np.all(branch.tlsPoints == branch[-1], axis=1)
                branch.tlsPoints = branch.tlsPoints[~isStartPoint & ~isEndPoint]
        return np.array_equal(pointSort(branch1.tlsPoints), pointSort(branch2.tlsPoints))
    else:
        return True


def subtrees_are_equal(stem1: Branch, stem2: Branch, compareTlsPoints: bool = True, atol: float = 0):
    if not branches_are_equal(stem1, stem2, compareTlsPoints, atol) or not len(stem1.children) == len(stem2.children):
        return False

    for child in stem1.children:
        is_in_other_subtree = False
        for other_child in stem2.children:
            if subtrees_are_equal(child, other_child, compareTlsPoints, atol):
                is_in_other_subtree = True
                break
        if not is_in_other_subtree:
            return False

    return True
