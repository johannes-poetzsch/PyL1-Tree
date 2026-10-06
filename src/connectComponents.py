from itertools import chain
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Iterable, List, Optional, Set, Tuple

from src.objects.branch import Branch
from src.objects.connectedComponent import ConnectedComponent
from src.objects.unconnectedComponent import UnconnectedComponent
from src.objects.connection import Connection
from src.objects.tlsPointManager import TlsPointManager
from src.utils.geometry import angle, normalizeVectors
from src.utils.iteration import breadthFirstSearch


def connectComponents(trunk: Branch,
                      componentStems: List[Branch],
                      unfixedPoints: npt.NDArray[np.double],
                      unassignedPoints: npt.NDArray[np.double],
                      minDistanceThreshold: float,
                      maxDistanceThreshold: float,
                      minAngleThreshold: float,
                      maxAngleThreshold: float,
                      resolution: float,
                      windowSize: float,
                      densityFactor: float,
                      stepSize: Optional[float] = None
                      ) -> Tuple[List[Branch], npt.NDArray[np.double]]:
    rootComponent = ConnectedComponent(list(breadthFirstSearch(trunk)))
    connectedComponents = {rootComponent}
    unconnectedComponents = {UnconnectedComponent(list(breadthFirstSearch(stem)))
                             for stem in componentStems if stem is not trunk}

    originalComponentCount = len(componentStems)
    branches = list(chain(breadthFirstSearch(trunk),
                          chain.from_iterable([c.branches for c in unconnectedComponents])))

    tlsManager = TlsPointManager(branches, unassignedPoints)

    unfixedLookUp, unfixedWrapper = KDTree(unfixedPoints), Branch(unfixedPoints)
    potentialConnections = generatePotentialConnections(branches, unfixedLookUp, unfixedWrapper, minDistanceThreshold)

    originalMaxDistanceThreshold = maxDistanceThreshold
    maxConnectionLength = 2.0

    while unconnectedComponents:
        for component in connectedComponents:
            if component.checkedDistanceThreshold < maxDistanceThreshold:
                for otherComponent in unconnectedComponents:
                    if component.getObjectBBDistance(otherComponent) <= maxDistanceThreshold:
                        findConnections(component, otherComponent, potentialConnections, tlsManager,
                                        minDistanceThreshold, maxDistanceThreshold,
                                        minAngleThreshold, maxAngleThreshold,
                                        windowSize, densityFactor, stepSize)
                component.checkedDistanceThreshold = maxDistanceThreshold

        connectableComponents = [c for c in unconnectedComponents if len(c.connections) > 0]
        if len(connectableComponents) > 0:
            connectionComponent = max(connectableComponents, key=lambda c: c.connections[0].score)
        else:
            if len(unconnectedComponents) <= originalComponentCount / 10:
                break
            elif maxDistanceThreshold + 0.2 <= maxConnectionLength:
                maxDistanceThreshold += 0.2
                continue
            else:
                connectionComponent = findContinuation(connectedComponents, unconnectedComponents, tlsManager,
                                                       maxConnectionLength)
                if connectionComponent is None:
                    break

        connectionBranch, splitEnd = connectComponent(connectionComponent, unconnectedComponents, connectedComponents,
                                                      tlsManager, resolution)

        updatePotentialConnections(potentialConnections, unfixedLookUp, unfixedWrapper, connectionBranch, splitEnd,
                                   minDistanceThreshold)

        maxDistanceThreshold = originalMaxDistanceThreshold

    trimmedPoints = tlsManager.releasePoints()
    for branch in reversed(list(breadthFirstSearch(trunk))):
        branch.removeTemporaryPoints()
        if len(branch.children) == 1:
            branch.expand()

    prunedBranches = list(chain.from_iterable([uc.branches for uc in unconnectedComponents]))

    return prunedBranches, trimmedPoints


def connectComponent(connectionComponent: UnconnectedComponent,
                     unconnectedComponents: Set[UnconnectedComponent],
                     connectedComponents: Set[ConnectedComponent],
                     tlsManager: TlsPointManager,
                     resolution: float,
                     ) -> Tuple[Branch, Optional[Branch]]:
    unconnectedComponents.remove(connectionComponent)
    connection = connectionComponent.connections[0]
    connectionComponent, connectionBranch, splitEnd = connectionComponent.connect(tlsManager, resolution)
    connectedComponents.add(connectionComponent)

    if splitEnd is not None:
        splitBranch = splitEnd.parent
        for component in connectedComponents:
            if splitBranch in component.directionData:
                del component.directionData[splitBranch]

    for component in unconnectedComponents:
        component.updateConnections(connection, connectionBranch)

    return connectionBranch, splitEnd


def findConnections(connectedComponent: ConnectedComponent,
                    unconnectedComponent: UnconnectedComponent,
                    potentialConnections: nx.Graph,
                    tlsManager: TlsPointManager,
                    minDistanceThreshold: float,
                    maxDistanceThreshold: float,
                    minAngleThreshold: float,
                    maxAngleThreshold: float,
                    windowSize: float,
                    densityFactor: float,
                    stepSize: Optional[float] = None):
    for endBranch in unconnectedComponent.branches:
        if connectedComponent.getObjectBBDistance(endBranch) > maxDistanceThreshold:
            continue
        directionsAtEnd = unconnectedComponent.getDirectionData(endBranch, minDistanceThreshold)

        for startBranch in connectedComponent.branches:
            if startBranch.getObjectBBDistance(endBranch) > maxDistanceThreshold:
                continue

            for endIdx, startDirection, childDirections in directionsAtEnd:
                for startIdx in range(1, len(startBranch)):
                    startPoint = startBranch[startIdx]
                    endPoint = endBranch[endIdx]
                    connectionLength = np.linalg.norm(endPoint - startPoint)

                    if (connectionLength > maxDistanceThreshold
                            or connectionLength <= connectedComponent.checkedDistanceThreshold):
                        continue

                    # score the direct connection between the points
                    connectionLine = np.stack((startPoint, endPoint))
                    connection = Connection(connectionLine, tlsManager,
                                            startBranch, startIdx, connectedComponent, endBranch, endIdx,
                                            startDirection, childDirections, minDistanceThreshold,
                                            minAngleThreshold, maxAngleThreshold, windowSize, densityFactor, stepSize)
                    if connection.isCredible:
                        unconnectedComponent.connections.append(connection)

                    try:
                        path = nx.shortest_path(potentialConnections, (startBranch, startIdx), (endBranch, endIdx),
                                                weight='distance')
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

                    connectionStrip = np.stack([branch.skeletonPoints[idx] for branch, idx in path])
                    connection = Connection(connectionStrip, tlsManager,
                                            startBranch, startIdx, connectedComponent, endBranch, endIdx,
                                            startDirection, childDirections, minDistanceThreshold,
                                            minAngleThreshold, maxAngleThreshold, windowSize, densityFactor)
                    if connection.isCredible:
                        unconnectedComponent.connections.append(connection)

    unconnectedComponent.connections.sort(key=lambda c: c.score, reverse=True)


def findContinuation(connectedComponents: Iterable[ConnectedComponent],
                     unconnectedComponents: Iterable[UnconnectedComponent],
                     tlsManager: TlsPointManager,
                     maxConnectionLength: float,
                     maxAngleThreshold: float = np.deg2rad(20)
                     ) -> Optional[UnconnectedComponent]:
    cBranches = list(chain.from_iterable([c.branches for c in connectedComponents]))
    cTerminalBranches = [b for b in cBranches if b.isLeaf()]
    # sort by descending radius
    def getRadius(branch: Branch) -> float:
        distances = tlsManager.getDistances(branch)
        return 0 if len(distances) == 0 else distances.mean()
    cRadii = {b: getRadius(b) for b in cTerminalBranches}
    cTerminalBranches.sort(key=lambda b: cRadii[b], reverse=True)

    ucBranches = list(chain.from_iterable([c.branches for c in unconnectedComponents]))
    ucLeaves = [b for b in ucBranches if b.isLeaf()]
    ucStems = [b for b in ucBranches if b.isStem()]
    ucTerminalBranches = ucLeaves + ucStems
    isLeaf = np.array([True, False]).repeat([len(ucLeaves), len(ucStems)])

    ucTerminalPoints = np.array([b[-1] for b in ucLeaves] + [b[0] for b in ucStems])
    ucTerminalLookUp = KDTree(ucTerminalPoints)

    for cBranch in cTerminalBranches:
        indices = np.array(ucTerminalLookUp.query_ball_point(cBranch[-1], maxConnectionLength))
        if len(indices) == 0:
            continue
        distances = np.linalg.norm(ucTerminalPoints[indices] - cBranch[-1], axis=-1)
        #order = np.flip(np.argsort([ucRadii[ucTerminalBranches[idx]] for idx in indices]))
        order = np.argsort(distances)

        for idx, d in zip(indices[order], distances[order]):
            ucBranch = ucTerminalBranches[idx]
            ucStartPoint = ucTerminalPoints[idx]

            cDirection = normalizeVectors(cBranch[-1] - cBranch[0])
            ucDirection = normalizeVectors(ucBranch[0] - ucBranch[-1] if isLeaf[idx] else ucBranch[-1] - ucBranch[0])

            if angle(cDirection, ucDirection) > maxAngleThreshold:
                continue

            connectionDirection = ucStartPoint - cBranch[-1]
            if (angle(connectionDirection, cDirection) > maxAngleThreshold
                    or angle(connectionDirection, ucDirection) > maxAngleThreshold):
                continue

            startComponent = [c for c in connectedComponents if cBranch in c.branches][0]
            endComponent = [uc for uc in unconnectedComponents if ucBranch in uc.branches][0]
            connection = Connection(np.array([cBranch[-1], ucStartPoint]), tlsManager,
                                    cBranch, len(cBranch) - 1, startComponent, ucBranch, len(ucBranch) - 1 if isLeaf[idx] else 0,
                                    cDirection, np.array([], dtype=np.double), maxConnectionLength, 0.0, 0.0, 1.0, 1.0)

            # use connection regardless of usual credibility
            endComponent.connections.append(connection)
            return endComponent
    return None


def generatePotentialConnections(branches: List[Branch],
                                 unfixedLookUp: KDTree,
                                 unfixedWrapper: Branch,
                                 minDistanceThreshold: float):
    unfixedPoints = unfixedLookUp.data
    potentialConnections = nx.Graph()

    unfixedNeighbors = unfixedLookUp.query_ball_tree(unfixedLookUp, minDistanceThreshold)
    for idx, neighborIndices in enumerate(unfixedNeighbors):
        for neighborIdx in neighborIndices:
            distance = np.linalg.norm(unfixedPoints[neighborIdx] - unfixedPoints[idx])
            potentialConnections.add_edge((unfixedWrapper, idx), (unfixedWrapper, neighborIdx), distance=distance)

    for branch in branches:
        for idx, point in enumerate(branch.skeletonPoints):
            potentialConnections.add_node((branch, idx))
            neighborIndices = unfixedLookUp.query_ball_point(point, minDistanceThreshold)
            for neighborIdx in neighborIndices:
                distance = np.linalg.norm(point - unfixedPoints[neighborIdx])
                potentialConnections.add_edge((branch, idx), (unfixedWrapper, neighborIdx), distance=distance)

    return potentialConnections


def updatePotentialConnections(potentialConnections: nx.Graph,
                               unfixedLookUp: KDTree,
                               unfixedWrapper: Branch,
                               newBranch: Branch,
                               splitEnd: Optional[Branch],
                               minDistanceThreshold: float):
    unfixedPoints = unfixedLookUp.data
    for idx, point in enumerate(newBranch.skeletonPoints):
        neighborIndices = unfixedLookUp.query_ball_point(point, minDistanceThreshold)
        for neighborIdx in neighborIndices:
            distance = np.linalg.norm(point - unfixedPoints[neighborIdx])
            potentialConnections.add_edge((newBranch, idx), (unfixedWrapper, neighborIdx), distance=distance)

    if splitEnd is not None:
        parent = splitEnd.parent
        nx.relabel_nodes(potentialConnections, {(parent, i + len(parent)-1): (splitEnd, i)
                                               for i in range(len(splitEnd))}, copy=False)
        sharedNode = (splitEnd, 0)
        duplicateNode = (parent, len(parent)-1)
        potentialConnections.add_node(duplicateNode)
        potentialConnections.add_weighted_edges_from([(duplicateNode, v, d) for u,v,d in
                                                     potentialConnections.edges(sharedNode, data="distance")],
                                                     weight="distance")
