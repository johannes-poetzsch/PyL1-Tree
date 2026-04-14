import numpy as np
import os
from typing import Optional

from src.objects.branch import Branch


def rootPath():
    return os.path.join(os.path.dirname(globals()['__file__']), os.pardir)


def import_branches(directory_path: str,
                    skeleton_points: np.ndarray,
                    tls_points: Optional[np.ndarray] = None,
                    end_point_ids: Optional[np.ndarray] = None):
    branches_path = os.path.join(rootPath(), directory_path, 'branches.txt')
    closest_branches_path = os.path.join(rootPath(), directory_path, 'closestBranches.txt')
    trimmed_path = os.path.join(rootPath(), directory_path, 'isTrimmed.txt')

    if not os.path.isfile(branches_path):
        raise FileNotFoundError

    if tls_points is not None and os.path.isfile(closest_branches_path):
        closest_branches = np.loadtxt(closest_branches_path, dtype=int) - 1 # subtract 1 to covert from 1- to 0-indexed
    else:
        closest_branches = None

    if closest_branches is not None and os.path.isfile(trimmed_path):
        trimmed = np.loadtxt(trimmed_path, dtype=bool)
        closest_branches[trimmed] = -1

    branches = []
    with open(branches_path) as input_file:
        for idx, line in enumerate(input_file):
            skeleton_ids = np.fromstring(line, dtype=int, sep=" ") - 1

            branch_tls_points = None if tls_points is None else tls_points[closest_branches == idx]

            branch = Branch(skeleton_points[skeleton_ids], branch_tls_points)

            if end_point_ids is not None:
                end_point_indices = np.flatnonzero(np.isin(skeleton_ids[1:-1], end_point_ids)) + 1
                branch.addEndPoints(end_point_indices)

            branches.append(branch)

    return branches
