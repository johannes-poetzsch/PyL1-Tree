import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Dict, Optional, Tuple

from src.objects.branch import Branch
from src.utils.iteration import breadthFirstSearch


def exportModel(stem: Branch,
                outputPath: str,
                branchRadii: Optional[Dict[Branch, float]] = None,
                radiusIsCredible: Optional[Dict[Branch, bool]] = None,
                exportTlsPoints: bool = True,
                unassignedTlsPoints: Optional[npt.NDArray[np.double]] = None,
                ):
    treeData = []

    for branch in breadthFirstSearch(stem):
        branchData = {
            "branchId": branch.id,
            "parentId": branch.parent.id if branch.parent is not None else np.nan,
            "x": branch.skeletonPoints[:,0],
            "y": branch.skeletonPoints[:,1],
            "z": branch.skeletonPoints[:,2],
            "cenOrder": branch.getCentrifugalOrder(),
        }

        if branchRadii is not None:
            branchData["radius"] = branchRadii[branch]
        if radiusIsCredible is not None:
            branchData["radiusIsCredible"] = radiusIsCredible[branch]

        treeData.append(pd.DataFrame(branchData))

    treeData = pd.concat(treeData, ignore_index=True)
    treeData.to_csv(outputPath, index=False)

    if exportTlsPoints:
        tlsData = []
        for branch in breadthFirstSearch(stem):
            tlsPoints = {
                "branchId": branch.id,
                "x": branch.skeletonPoints[:,0],
                "y": branch.skeletonPoints[:,1],
                "z": branch.skeletonPoints[:,2],}
            tlsData.append(pd.DataFrame(tlsPoints))

        if unassignedTlsPoints is not None:
            tlsPoints = {
                "branchId": np.nan,
                "x": unassignedTlsPoints[:,0],
                "y": unassignedTlsPoints[:,1],
                "z": unassignedTlsPoints[:,2],
            }
            tlsData.append(pd.DataFrame(tlsPoints))
        tlsData = pd.concat(tlsData, ignore_index=True)
        tlsData.to_csv(outputPath + "_tlsPoints", index=False)


def importModel(inputPath: str) -> Tuple[Branch, Optional[npt.NDArray[np.double]]]:
    stem = None
    unassignedTlsPoints = None
    branches = {}

    treeData = pd.read_csv(inputPath)
    branchIds = treeData["branchId"].unique()
    branchIndices = treeData.groupby("branchId", sort=False).groups

    for idx, branchId in enumerate(branchIds):
        branchData = treeData[branchIndices[branchId]]

        skeletonPoints = np.array(branchData[["x", "y", "z"]], dtype=np.double)

        parentId = branchData["parentId"][0]
        parent = branches[parentId]

        branch = Branch(skeletonPoints, parent=parent)
        if idx == 0:
            stem = branch

    try:
        tlsData = pd.read_csv(inputPath + "_tlsPoints")
    except FileNotFoundError:
        return stem, None

    branchIds = tlsData["branchId"].unique()
    branchIndices = tlsData.groupby("branchId", sort=False).groups

    for branchId in branchIds:
        tlsPoints = tlsData[branchIndices[branchId]][["x", "y", "z"]]
        if np.isnan(branchId):
            unassignedTlsPoints = tlsPoints
        else:
            branches[branchId].tlsPoints = tlsPoints

    return stem, unassignedTlsPoints
