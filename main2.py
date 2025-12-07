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
    heightAttributeIdx = 2
    tlsPoints = tlsPoints[np.argsort(tlsPoints[:,heightAttributeIdx])] # sort by height
    seedPoints, relPathPositions = generateSeeds(tlsPoints, parameters)
    branches = generateSkeleton(tlsPoints, seedPoints, relPathPositions, parameters)
    


if __name__ == '__main__':
    main()
