import numpy as np
import yaml

from src.adjustBifurcations import adjustBifurcations
from src.constructTreeGraph import constructTreeGraph
from src.constructTreeModel import constructTreeModel
from src.extendSkeleton import extendSkeleton
from src.filterBranches import filterTreeBranches
from src.L1MedianAlgorithm.generateSeeds import generateSeeds
from src.L1MedianAlgorithm.generateBranches import generateBranches
from src.smoothBranches import smoothBranches
from src.utils.ioUtils import exportModel


def main():
    with open("Data/Tree_1.txt") as inputFile:
        tlsPoints = np.loadtxt(inputFile, dtype=np.double)
    with open("parameters.yaml") as config:
        parameters = yaml.safe_load(config)["parameters"]

    if parameters["heightAttributeIdx"] != 2:
        h = parameters["heightAttributeIdx"]
        height = tlsPoints[:,h].copy()
        tlsPoints[:,h] = tlsPoints[:,2]
        tlsPoints[:,2] = height

    seedPoints, relPathPositions, tlsPoints = generateSeeds(tlsPoints, parameters)
    branches = generateBranches(tlsPoints, seedPoints, relPathPositions, parameters)
    stem, prunedTlsPoints1 = constructTreeGraph(tlsPoints, branches, parameters)

    adjustBifurcations(stem, parameters)

    prunedTlsPoints2 = filterTreeBranches(stem, parameters)

    prunedTlsPoints = np.concatenate((prunedTlsPoints1, prunedTlsPoints2), axis=0)

    unassignedTlsPoints = smoothBranches(stem, tlsPoints, prunedTlsPoints, parameters)
    extendSkeleton(stem, unassignedTlsPoints, parameters)

    branchRadii, radiusIsCredible = constructTreeModel(stem)
    exportModel(stem, branchRadii, radiusIsCredible, "Results/Tree_1.csv")

    print("finished")
    


if __name__ == '__main__':
    main()
