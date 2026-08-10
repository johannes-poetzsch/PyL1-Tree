from typing import Dict, List

import numpy as np
import numpy.typing as npt

from src.objects.branch import Branch
from src.utils.geometry import findClosestCylinders
from src.utils.iteration import breadthFirstSearch


def calculateMetrics(cylinders: npt.NDArray, tlsPoints: npt.NDArray[np.double]) -> Dict[Branch, float]:
    trunkCylinders = cylinders[cylinders["branch_order_gaaliche"] == 0]

    surfaceDistances = calculateSurfaceDistances(cylinders, tlsPoints)
    totalVolume = getVolumes(cylinders).sum()
    trunkVolume = getVolumes(trunkCylinders).sum()
    trunkLength = getLengths(trunkCylinders).sum()

    accuracy = surfaceDistances.mean()
    standardDeviation = surfaceDistances.std()

    groundHeight = cylinders["start"][0, 2]
    treeHeight = cylinders["end"][2].max() - groundHeight

    dbs = getTrunkRadiusAtHeight(cylinders, 1.3, groundHeight, treeHeight)


def calculateSurfaceDistances(cylinders: npt.NDArray, tlsPoints: npt.NDArray[np.double]):
    _, cylinderDistances = findClosestCylinders(cylinders["start"], cylinders["end"], cylinders["radius"], tlsPoints)

    return cylinderDistances


def getLengths(cylinders: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    lengths = np.linalg.norm(cylinders["start"] - cylinders["end"], axis=1)
    return lengths


def getTrunkRadiusAtHeight(trunkCylinders, height, groundHeight, treeHeight):
    if height > treeHeight:
        return 0

    height = height - groundHeight
    isAtHeight = trunkCylinders["start"] <= height & trunkCylinders["end"] >= height

    radius = trunkCylinders["radius"][isAtHeight].squeeze()
    return radius


def getVolumes(cylinders):
    volumes = np.pi * cylinders["radius"]**2 * getLengths(cylinders)
    return volumes