import numpy as np

from src.objects.branch import Branch


def branches_are_equal(branch1: Branch, branch2: Branch, compareTlsPoints: bool = True):
    if not np.array_equal(branch1.skeletonPoints, branch2.skeletonPoints):
        return False

    if compareTlsPoints:
        return np.array_equal(branch1.tlsPoints.sort(axis=0), branch2.tlsPoints.sort(axis=0))
    else:
        return True


def subtrees_are_equal(stem1: Branch, stem2: Branch, compareTlsPoints: bool = True):
    if not branches_are_equal(stem1, stem2, compareTlsPoints) or not len(stem1.children) == len(stem2.children):
        return False

    for child in stem1.children:
        is_in_other_subtree = False
        for other_child in stem2.children:
            if subtrees_are_equal(child, other_child, compareTlsPoints):
                is_in_other_subtree = True
                break
        if not is_in_other_subtree:
            return False

    return True
