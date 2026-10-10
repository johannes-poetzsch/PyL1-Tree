import argparse
import sys
import time
from os import path
from typing import Any, Dict

import laspy
import numpy as np
import numpy.typing as npt
import yaml

from src.adjustBifurcations import adjustBifurcations
from src.constructTreeGraph import constructTreeGraph
from src.constructTreeModel import constructTreeModel
from src.extendSkeleton import extendSkeleton
from src.filterBranches import filterByLength, filterTreeBranches
from src.L1MedianAlgorithm.generateSeeds import generateSeeds
from src.L1MedianAlgorithm.generateBranches import generateBranches
from src.smoothBranches import cleanUpBranches, smoothBranches
from src.utils.geometry import setFastStripPointDistances
from src.utils.iteration import breadthFirstSearch
from src.utils.voxelDownsampling import voxelDownsampling


def main():
    args = parseArguments(sys.argv)
    inputFile = args.inputFile
    fileName, extension = path.splitext(path.basename(inputFile))
    if not path.isfile(inputFile):
        print(f"Input file \"{inputFile}\" does not exist")
        sys.exit(1)

    if extension == ".txt":
        with open(inputFile) as file:
            tlsPoints = np.loadtxt(file, dtype=np.double, delimiter=args.delimiter, comments=["\"", "#", "/"],
                                   usecols=(1,2,3) if hasIndexColumn(inputFile, args.delimiter) else (0,1,2))
    elif extension == ".las" or extension == ".laz":
        with laspy.open(inputFile) as file:
            las = file.read()
            tlsPoints = np.vstack((las.x, las.y, las.z), dtype=np.double).transpose()
    else:
        print(f"File extension \"{extension}\" not supported")
        sys.exit(1)


    with open(args.parameters) as paramFile:
        parameters = yaml.safe_load(paramFile)["parameters"]

    print(f"creating QSM for {fileName}")
    cylinders = createQSM(tlsPoints, parameters, args.workers)
    if cylinders is None:
        return
    outputFile = path.join(args.outputDirectory, fileName)
    np.save(outputFile, cylinders)
    print("saved cylinders to {}".format(outputFile))
    print()


def hasIndexColumn(file, delimiter):
    with open(file) as f:
        firstColumn = np.loadtxt(file, dtype=float, delimiter=delimiter, usecols=0, comments=["\"", "#", "/"])
    idx = np.arange(len(firstColumn))
    return np.all(idx == firstColumn) or np.all(idx+1 == firstColumn)


def parseArguments(argv):
    parser = argparse.ArgumentParser(description='Python implementation of the L1Tree algorithm from the paper:\n'
                                                 'Yuhao Feng, Yanjun Su*, Jiatong Wang, Jiabo Yan, Xiaotian Qi, Eduardo Eiji Maeda, Matheus Henrique Nunes, Xiaoxia Zhao, Xiaoqiang Liu, Xiaoyong Wu, Chen Yang, Jiamin Pan, Kai Dong, Danhua Zhang, Tianyu Hu, Jingyun Fang*. 2024. L1-Tree: A novel algorithm for constructing 3D tree models and estimating branch architectural traits using terrestrial laser scanning data. Remote Sensing of Environment 314, 114390. https://doi.org/10.1016/j.rse.2024.114390')

    parser.add_argument('inputFile', type=str, help='Path to the input file')
    parser.add_argument('outputDirectory', type=str,
                        help="Path to the output directory. The file name will be the same as the input file",
                        default=None)
    parser.add_argument("-p", "--parameters", type=str, help="Parameter file",
                        default=path.join(path.dirname(argv[0]), "parameters.yaml"))
    parser.add_argument("--delimiter", type=str, help="Delimiter used in the input point clouds", default=None)
    parser.add_argument("-w", "--workers", type=int, help="Number of worker threads", default=1)

    return parser.parse_args(argv[1:])


def createQSM(tlsPoints: npt.NDArray[np.double],
              parameters: Dict[str, Any],
              workers: int = 1):
    setFastStripPointDistances(parameters.get("useFastStripPointDistances", True))

    if parameters["heightAttributeIdx"] != 2:
        h = parameters["heightAttributeIdx"]
        height = tlsPoints[:,h].copy()
        tlsPoints[:,h] = tlsPoints[:,2]
        tlsPoints[:,2] = height

    start_time = time.time()

    tlsPoints, _, _ = voxelDownsampling(tlsPoints, 0.002)

    print("original point count:", len(tlsPoints))
    print(f"generating seeds (starting after {time.time()-start_time} s)")
    seedPoints, relPathPositions, tlsPoints = generateSeeds(tlsPoints, parameters, useStumpRemoval=True, useRansac=False)
    print("base radius:", parameters["baseRadius"])
    print("remaining point count:", len(tlsPoints))

    print(f"generating branches (starting after {time.time()-start_time} s)")
    branches, unfixedPoints = generateBranches(tlsPoints, seedPoints, relPathPositions, parameters, workers)
    print("number of initial branches:", len(branches))

    print(f"constructing initial tree graph (starting after {time.time()-start_time} s)")
    trunk, prunedTlsPoints1 = constructTreeGraph(tlsPoints, branches, unfixedPoints, parameters)
    print("Number of Pruned TlsPoints:", len(prunedTlsPoints1))
    print("Number of connected branches:", len(list(breadthFirstSearch(trunk))))
    print("max Height:", max(np.concatenate([b.skeletonPoints[:,2] for b in breadthFirstSearch(trunk)], axis=0)))

    print(f"adjusting bifurcations (starting after {time.time()-start_time} s)")
    adjustBifurcations(trunk, parameters)
    prunedTlsPoints2 = filterTreeBranches(trunk, parameters)
    prunedTlsPoints = np.concatenate((prunedTlsPoints1, prunedTlsPoints2), axis=0)

    print(f"smoothing branches (starting after {time.time()-start_time} s)")
    unassignedTlsPoints = smoothBranches(trunk, tlsPoints, prunedTlsPoints, parameters, workers)

    #if len(unassignedTlsPoints) < len(tlsPoints) / 2:
    print(f"extending skeleton (starting after {time.time()-start_time} s)")
    extendSkeleton(trunk, unassignedTlsPoints, parameters)

    print(f"adjusting bifurcations (starting after {time.time()-start_time} s)")
    adjustBifurcations(trunk, parameters)
    filterByLength(trunk, parameters["branchSizeThreshold"], parameters["resolution"])
    cleanUpBranches(trunk, unassignedTlsPoints)
    #else:
    #    warnings.warn("Most of the point cloud could not be modeled.")

    print(f"constructing tree model (starting after {time.time()-start_time} s)")
    cylinders = constructTreeModel(trunk)
    #exportModel(trunk, branchRadii, radiusIsCredible, "Results/Tree_1.csv")
    print(f"total time: {time.time()-start_time} s")
    return cylinders


if __name__ == '__main__':
    main()
