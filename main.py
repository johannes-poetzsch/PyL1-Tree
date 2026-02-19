import matlab.engine
import numpy as np
from typing import List

from src.adjustBifurcations import adjustBifurcations
from src.extendSkeleton import extendSkeleton
from src.filterBranches import filterTreeBranches
from src.smoothBranches import smoothBranches
from src.objects.branch import Branch
from src.utils.iteration import breadthFirstSearch
from src.utils.parseMatlab import parse_struct, import_tree_model


def main():
    eng = matlab.engine.start_matlab()
    eng.eval("load(\"L1TreeAdjustBifurcation_Input.mat\")", nargout=0)
    
    treeModel, unassignedTlsPoints = import_tree_model(eng)
    stem = treeModel[0]
    parameters = parse_struct(eng, "parameters")
    #tlsPoints = cp.array(eng.eval("tlsPoints"))  # TODO branchIDs -1
    eng.exit()

    print("finished import")

    adjustBifurcations(stem, parameters)
    
    print("finished adjusting bifurcations")
    treeModel = list(breadthFirstSearch(stem))
    print(len(treeModel))

    prunedTlsPoints = filterTreeBranches(stem, parameters)

    if prunedTlsPoints is not None:
        unassignedTlsPoints = np.append(unassignedTlsPoints, prunedTlsPoints, axis=0)

    unassignedTlsPoints = smoothBranches(stem, tlsPoints, unassignedTlsPoints, parameters)
    extendSkeleton(stem, unassignedTlsPoints, parameters)

    print("finished")


if __name__ == '__main__':
    main()
