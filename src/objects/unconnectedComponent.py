import numpy as np
from typing import List, Optional, Tuple

from src.objects.branch import Branch
from src.objects.component import Component
from src.objects.connectedComponent import ConnectedComponent
from src.objects.connection import Connection
from src.objects.tlsPointManager import TlsPointManager
from src.utils.geometry import normalizeVectors
from src.utils.iteration import breadthFirstSearch


class UnconnectedComponent(Component):
    def __init__(self, branches: List[Branch]):
        Component.__init__(self, branches)
        self.connections: List[Connection] = []
        self.directionData = {}


    def connect(self,
                tlsManager: TlsPointManager,
                resolution: float) -> Tuple[ConnectedComponent, Branch, Optional[Branch]]:
        if len(self.connections) == 0:
            raise ValueError("Components can not be connected.")
        connection = self.connections[0]
        connectionBranch, sibling = connection.apply(resolution)

        misorientedChild, newChild = tlsManager.splitBranch(connection.endBranch, connection.endIdx)
        if misorientedChild is None:
            connection.endBranch.setParent(connectionBranch)
        else:
            misorientedChild.reorientSubtree(connectionBranch)

        tlsManager.expandBranch(connectionBranch)

        connectedComponent = ConnectedComponent(list(breadthFirstSearch(connectionBranch)))

        return connectedComponent, connectionBranch, sibling


    def generateDirectionData(self, branch: Branch, minDistanceThreshold: float):
        self.directionData[branch] = []
        connectionStarts = []

        for startIdx, endIdx in UnconnectedComponent.getBranchSegments(branch):
            startDirection = normalizeVectors(branch[startIdx] - branch[endIdx])
            connectionStarts.append((startIdx, startDirection))
            connectionStarts.append((endIdx, -startDirection))

        for idx, startDirection in connectionStarts:
            firstHalfDirections = self.getDirections(idx, -1, branch, minDistanceThreshold)
            secondHalfDirections = self.getDirections(idx, 1, branch, minDistanceThreshold)
            directions = firstHalfDirections + secondHalfDirections

            self.directionData[branch].append((idx, startDirection, directions))


    def updateConnections(self, realizedConnection: Connection, connectionBranch: Branch):
        for connection in self.connections:
            connection.update(realizedConnection, connectionBranch)

        self.connections = [c for c in self.connections if c.isCredible]
        self.connections.sort(key=lambda connection: connection.score, reverse=True)


    # Each branch in a component was created by fusing branch segments at their end points.
    # IsEndPoint returns whether a skeleton point in a branch was such an end point.
    # The startDirection is the direction of the branch segment starting from an end point.
    # Since branches share points with their parent, the first (end) point in a branch can either belong
    # to the first branch segment of that branch or to the last branch segment of its parent.
    # Depending on this, the first branch segment either starts at the first or second branch point.
    @staticmethod
    def getBranchSegments(branch: Branch):
        isEndPoint = branch.isEndPoint(np.arange(len(branch), dtype=int))
        endPointIndices = np.flatnonzero(isEndPoint)

        if branch.isStem(): # stems must start with a branch segment
            firstStart = 0
        elif branch.isLeaf(): # leafs must end with a branch segment
            firstStart = len(endPointIndices) % 2
        elif len(endPointIndices) < len(branch):
            nonEndPointIndices = np.flatnonzero(np.logical_not(isEndPoint))
            firstStart = 1 - nonEndPointIndices[0] % 2
        else:
            parentSegments = UnconnectedComponent.getBranchSegments(branch.parent)
            if len(parentSegments) == 0:
                firstStart = 0
            else:
                parentEndsInSegment = parentSegments[-1][1] == len(branch.parent) - 1
                firstStart = 1 if parentEndsInSegment else 0

        segments = [(endPointIndices[i], endPointIndices[i+1]) for i in range(firstStart, len(endPointIndices)-1, 2)]
        return segments
