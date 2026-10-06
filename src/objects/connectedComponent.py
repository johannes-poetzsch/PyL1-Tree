from typing import List

from src.objects.branch import Branch
from src.objects.component import Component


class ConnectedComponent(Component):
    def __init__(self, branches: List[Branch]):
        Component.__init__(self, branches)
        self.checkedDistanceThreshold = 0.0


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        self.directionData[branch] = [[]]
        maxIdx = len(branch) - 1

        # if the branch is the trunk connecting to idx 0 would not leave a clear trunk
        # if the start branch is not the trunk, idx 0 is handled as the maxIdx of the parent
        for idx in range(1, maxIdx):
            prePointDirections = Component.getDirections(idx, -1, branch, minDistanceThreshold)
            postPointDirections = Component.getDirections(idx, 1, branch, minDistanceThreshold)

            self.directionData[branch].append((prePointDirections, postPointDirections))

        # tip point of branch
        prePointDirections = Component.getDirections(maxIdx, -1, branch, minDistanceThreshold)
        postPointDirections = []
        for child in branch.children:
            postPointDirections.extend(Component.getDirections(0, 1, child, minDistanceThreshold))

        self.directionData[branch].append((prePointDirections, postPointDirections))
