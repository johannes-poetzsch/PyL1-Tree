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

    tlsPoints = np.unique(tlsPoints, axis=0)
    parameters["heightAttributeIdx"] = 2 # TODO: ensure height attribute is Z, reorder otherwise
    tlsPoints = tlsPoints[np.argsort(tlsPoints[:, parameters["heightAttributeIdx"]])]
    seedPoints, relPathPositions = generateSeeds(tlsPoints, parameters)
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
