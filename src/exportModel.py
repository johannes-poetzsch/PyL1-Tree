from os import PathLike

import numpy as np
import pandas as pd
from typing import Dict

from objects.branch import Branch
from utils.iteration import breadthFirstSearch


def exportModel(stem: Branch,
                branchRadii: Dict[Branch, float],
                radiusIsCredible: Dict[Branch, bool],
                outputPath: str | PathLike):
    branchData = []
    for branch in breadthFirstSearch(stem):
        pointData = {
            "branchId": branch.id,
            "parentId": branch.parent.id if branch.parent is not None else np.nan,
            "x": branch.skeletonPoints[:,0],
            "y": branch.skeletonPoints[:,1],
            "z": branch.skeletonPoints[:,2],
            "radius": branchRadii[branch],
            "radiusIsCredible": radiusIsCredible[branch],
            "cenOrder": branch.getCentrifugalOrder(),
        }

        branchData.append(pd.DataFrame(pointData))

    branchData = pd.concat(branchData, ignore_index=True)
    branchData.to_csv(outputPath, index=False)
