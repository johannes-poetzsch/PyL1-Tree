import argparse
import json
import sys
import time
from datetime import datetime
from os import path
from typing import Any, Dict

import laspy
import numpy as np
import numpy.typing as npt
import pandas as pd
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
    analytics: Dict[str, Any] = {"run_started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                 "input_name": path.basename(inputFile), "input_path": path.abspath(inputFile),
                                 "workers": args.workers, "point_count_input": len(tlsPoints)}
    cylinders = createQSM(tlsPoints, parameters, args.workers, analytics, args.useMultiprocessing)
    addParametersToAnalytics(analytics, parameters)
    analyticsFile = path.join(args.outputDirectory, "analytics.csv")
    appendAnalyticsRow(analyticsFile, analytics)
    print(f"saved analytics to {analyticsFile}")
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


def addParametersToAnalytics(analytics: Dict[str, Any], parameters: Dict[str, Any]):
    def normalizeValue(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: normalizeValue(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalizeValue(item) for item in value]
        return value

    for name, value in parameters.items():
        value = normalizeValue(value)
        analytics[f"parameter_{name}"] = json.dumps(value) if isinstance(value, (dict, list)) else value


def appendAnalyticsRow(analyticsFile: str, analytics: Dict[str, Any]):
    newRow = pd.DataFrame([analytics])
    if not path.isfile(analyticsFile):
        newRow.to_csv(analyticsFile, index=False)
        return

    existingColumns = pd.read_csv(analyticsFile, nrows=0).columns.tolist()
    if all(column in existingColumns for column in newRow.columns):
        newRow.reindex(columns=existingColumns).to_csv(analyticsFile, mode="a", header=False, index=False)
        return

    # Preserve earlier runs while expanding the CSV when new analytics columns are added.
    existingRows = pd.read_csv(analyticsFile)
    pd.concat([existingRows, newRow], ignore_index=True).to_csv(analyticsFile, index=False)


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
    parser.add_argument("-w", "--workers", type=int, help="Number of worker processes", default=1)
    parser.add_argument("--no-multiprocessing", dest="useMultiprocessing", action="store_false",
                        help="Run parallel-capable stages in the current process")

    return parser.parse_args(argv[1:])


def createQSM(tlsPoints: npt.NDArray[np.double],
              parameters: Dict[str, Any],
              workers: int = 1,
              analytics: Dict[str, Any] | None = None,
              useMultiprocessing: bool = True):
    if analytics is None:
        analytics = {}

    setFastStripPointDistances(parameters.get("useFastStripPointDistances", True))

    if parameters["heightAttributeIdx"] != 2:
        h = parameters["heightAttributeIdx"]
        height = tlsPoints[:,h].copy()
        tlsPoints[:,h] = tlsPoints[:,2]
        tlsPoints[:,2] = height

    start_time = time.perf_counter()

    stageStart = time.perf_counter()
    tlsPoints, _, _ = voxelDownsampling(tlsPoints, 0.002)
    analytics["point_count_after_voxel_downsampling"] = len(tlsPoints)
    analytics["time_voxel_downsampling_s"] = time.perf_counter() - stageStart

    print("point count after voxel downsampling:", len(tlsPoints))
    print(f"generating seeds (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    seedPoints, relPathPositions, tlsPoints = generateSeeds(tlsPoints, parameters, useStumpRemoval=True, useRansac=False,
                                                            analytics=analytics)
    analytics["point_count_after_seed_generation"] = len(tlsPoints)
    analytics["seed_point_count"] = len(seedPoints)
    analytics["time_seed_generation_s"] = time.perf_counter() - stageStart
    print("base radius:", parameters["baseRadius"])
    print("remaining point count:", len(tlsPoints))

    print(f"generating branches (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    branches, unfixedPoints = generateBranches(tlsPoints, seedPoints, relPathPositions, parameters, workers,
                                               useMultiprocessing)
    analytics["time_branch_generation_s"] = time.perf_counter() - stageStart
    analytics["branch_count_after_generation"] = len(branches)
    print("number of initial branches:", len(branches))

    print(f"constructing initial tree graph (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    trunk, prunedTlsPoints1 = constructTreeGraph(tlsPoints, branches, unfixedPoints, parameters,
                                                 analytics=analytics, workers=workers,
                                                 useMultiprocessing=useMultiprocessing)
    analytics["time_initial_tree_graph_construction_s"] = time.perf_counter() - stageStart
    analytics["point_count_pruned_during_tree_graph_construction"] = len(prunedTlsPoints1)
    analytics["point_count_after_tree_graph_filtering"] = sum(len(b.tlsPoints) for b in breadthFirstSearch(trunk))
    analytics["branch_count_after_initial_graph_filtering"] = len(list(breadthFirstSearch(trunk)))
    print("Number of Pruned TlsPoints:", len(prunedTlsPoints1))
    print("Number of connected branches:", len(list(breadthFirstSearch(trunk))))
    print("max Height:", max(np.concatenate([b.skeletonPoints[:,2] for b in breadthFirstSearch(trunk)], axis=0)))

    print(f"adjusting bifurcations (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    adjustBifurcations(trunk, parameters)
    analytics["time_bifurcation_adjustment_initial_s"] = time.perf_counter() - stageStart
    print(f"filtering branches (starting after {time.perf_counter() - start_time} s)")
    stageStart = time.perf_counter()
    prunedTlsPoints2 = filterTreeBranches(trunk, parameters)
    analytics["time_post_adjustment_branch_filtering_s"] = time.perf_counter() - stageStart
    analytics["point_count_pruned_by_post_adjustment_filtering"] = len(prunedTlsPoints2)
    prunedTlsPoints = np.concatenate((prunedTlsPoints1, prunedTlsPoints2), axis=0)

    print(f"smoothing branches (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    unassignedTlsPoints = smoothBranches(trunk, tlsPoints, prunedTlsPoints, parameters, workers,
                                         useMultiprocessing)
    analytics["time_branch_smoothing_s"] = time.perf_counter() - stageStart
    analytics["point_count_clustering_input"] = len(unassignedTlsPoints)

    print(f"extending skeleton (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    extendSkeleton(trunk, unassignedTlsPoints, parameters, analytics=analytics)
    analytics["time_skeleton_extension_s"] = time.perf_counter() - stageStart
    analytics["time_skeleton_extension_without_dbscan_s"] = (
        analytics["time_skeleton_extension_s"] - analytics["time_dbscan_clustering_s"])

    print(f"adjusting bifurcations (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    adjustBifurcations(trunk, parameters)
    analytics["time_bifurcation_adjustment_final_s"] = time.perf_counter() - stageStart
    print(f"filtering branches (starting after {time.perf_counter() - start_time} s)")
    stageStart = time.perf_counter()
    filterByLength(trunk, parameters["branchSizeThreshold"], parameters["resolution"])
    analytics["time_final_length_filtering_s"] = time.perf_counter() - stageStart
    print(f"cleaning up branches (starting after {time.perf_counter() - start_time} s)")
    stageStart = time.perf_counter()
    cleanUpBranches(trunk, unassignedTlsPoints)
    analytics["time_final_branch_cleanup_s"] = time.perf_counter() - stageStart
    analytics["branch_count_final"] = len(list(breadthFirstSearch(trunk)))

    print(f"constructing tree model (starting after {time.perf_counter()-start_time} s)")
    stageStart = time.perf_counter()
    cylinders = constructTreeModel(trunk)
    analytics["time_tree_model_construction_s"] = time.perf_counter() - stageStart
    analytics["time_total_s"] = time.perf_counter()-start_time
    print(f"total time: {analytics['time_total_s']} s")
    return cylinders


if __name__ == '__main__':
    main()
