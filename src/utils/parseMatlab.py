import matlab.engine as m
import cupy as cp
import numpy.typing as npt
import numpy as np

from typing import List, Tuple
from ..objects.branch import Branch

def parse_struct_array(eng: m.MatlabEngine,
                       name: str):
    branch_count = int(eng.eval(f"length({name})"))
    keys = eng.eval(f"fieldnames({name})")
    treeModel = []

    for i in range(branch_count):
        treeModel.append(parse_struct(eng, f"{name}({i+1})", keys))
    return treeModel


def parse_struct(eng: m.MatlabEngine,
                 name: str,
                 keys=None):
    if keys is None:
        keys = eng.eval(f"fieldnames({name})")

    dict = {}
    for key in keys:
        value = f"{name}.{key}"
        if eng.eval(f"isscalar({value}) | ischar({value})"):
            dict[key] = eng.eval(value)
        else:
            dict[key] = np.array(eng.eval(value))
    return dict

def import_tree_model(eng: m.MatlabEngine) -> List[Branch]:
    tlsBranchIDs = np.array(eng.eval("tlsPoints(:,4)"), dtype=np.intp).reshape(-1) - 1
    tlsPoints = np.array(eng.eval("tlsPoints(:,1:3)"), dtype=np.double)

    treeModel = []
    test0 = tlsPoints[tlsBranchIDs == 0]
    test1 = tlsPoints[tlsBranchIDs == -1]
    for branchIndex in range(int(eng.eval("length(treeModel)"))):
        skeletonIDs = np.array(eng.eval(f"treeModel({branchIndex+1}).SkeletonIDs"), dtype=np.intp).flatten() - 1
        skeletonPoints = np.array(eng.eval(f"treeModel({branchIndex+1}).Curve"), dtype=np.double)
        cenOrder = int(eng.eval(f"treeModel({branchIndex+1}).CenOrder")) - 1

        treeModel.append(Branch(branchIndex, skeletonIDs, skeletonPoints, tlsPoints[tlsBranchIDs == branchIndex], cenOrder))

    return treeModel

