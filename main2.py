import numpy as np
import yaml

from src.L1MedianAlgorithm.generateSeeds import generateSeeds
from src.L1MedianAlgorithm.generateSkeleton import generateSkeleton


def main():
    with open("Data/Tree_1.txt") as input:
        tlsPoints = np.loadtxt(input, dtype=np.double)
    with open("parameters.yaml") as config:
        parameters = yaml.safe_load(config)["parameters"]

    tlsPoints = np.unique(tlsPoints, axis=0)
    parameters["heightAttributeIdx"] = 2 # TODO: ensure height attribute is Z, reorder otherwise
    tlsPoints = tlsPoints[np.argsort(tlsPoints[:, parameters["heightAttributeIdx"]])]
    seedPoints, relPathPositions = generateSeeds(tlsPoints, parameters)
    branches = generateSkeleton(tlsPoints, seedPoints, relPathPositions, parameters)

    return
    


if __name__ == '__main__':
    main()
