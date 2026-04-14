from itertools import chain
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing import Dict, List, Optional, Set, Tuple

from src.objects.branch import Branch
from src.objects.component import ConnectedComponent, UnconnectedComponent
from src.scoreConnection import scoreConnection
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
                      densityFactor: float
                      ) -> Tuple[List[Branch], npt.NDArray[np.double]]:
    unconnectedComponents = {UnconnectedComponent(list(breadthFirstSearch(stem)))
                             for stem in componentStems if stem is not trunk}

    originalComponentCount = len(componentStems)

    allBranches: Set[Branch] = set(chain(breadthFirstSearch(trunk),
                                         chain.from_iterable([c.branches for c in unconnectedComponents])))

    unfixedLookUp, unfixedWrapper = KDTree(unfixedPoints), Branch(unfixedPoints)
    potentialConnections = generatePotentialConnections(allBranches, unfixedLookUp, unfixedWrapper, minDistanceThreshold)

    assignedPoints = np.concatenate([b.tlsPoints for b in allBranches], axis=0)
    assignedPointsDistances = np.concatenate([b.getDistances(b.tlsPoints) for b in allBranches], axis=0)
    assignedPointsLookUp = KDTree(assignedPoints)

    unassignedPointsDistances = Branch.findMinBranchDistances(unassignedPoints, allBranches)
    unassignedPointsLookUp = KDTree(unassignedPoints)

    rootComponent = ConnectedComponent(list(breadthFirstSearch(trunk)),
                                       assignedPointsLookUp, assignedPointsDistances)
    connectedComponents = {rootComponent}

    originalMaxDistanceThreshold = maxDistanceThreshold
    maxConnectionLength = 2.0
    originalBranchIndices = {}

    while unconnectedComponents:
        for component in connectedComponents:
            if component.checkedDistanceThreshold < maxDistanceThreshold:
                for otherComponent in unconnectedComponents:
                    if component.getBoundingBoxDistance(otherComponent) <= maxDistanceThreshold:
                        findConnections(component, otherComponent, potentialConnections,
                                        unassignedPointsLookUp, unassignedPointsDistances,
                                        minDistanceThreshold, maxDistanceThreshold,
                                        minAngleThreshold, maxAngleThreshold,
                                        windowSize, densityFactor)
                component.checkedDistanceThreshold = maxDistanceThreshold

        connectableComponents = [c for c in unconnectedComponents if len(c.connections) > 0]

        if len(connectableComponents) > 0:
            connectionBranch, splitEnd = connectComponent(max(connectableComponents, key=lambda c: c.connections[0]),
                                                          unconnectedComponents, connectedComponents, allBranches,
                                                          assignedPointsLookUp, assignedPointsDistances,
                                                          originalBranchIndices, resolution,
                                                          maxConnectionLength, maxAngleThreshold, maxDistanceThreshold)

            updateTlsData(allBranches)

            updatePotentialConnections(potentialConnections, unfixedLookUp, unfixedWrapper, connectionBranch, splitEnd,
                                       minDistanceThreshold)

            maxDistanceThreshold = originalMaxDistanceThreshold
        else:
            if (maxDistanceThreshold + 0.2 > maxConnectionLength
                    or len(unconnectedComponents) <= originalComponentCount / 10):
                break
            maxDistanceThreshold += 0.2

    for branch, originalIndices in originalBranchIndices.items():
        branch.skeletonPoints = branch.skeletonPoints[originalIndices]

    treeBranches = list(breadthFirstSearch(trunk))
    prunedBranches = list(chain.from_iterable([uc.branches for uc in unconnectedComponents]))
    allBranches = list(chain(treeBranches, prunedBranches))

    trimmedPoints1 = Branch.reassignPoints(treeBranches, useTrimming=True)
    trimmedPoints2 = Branch.assignAllPoints(unassignedPoints, allBranches, useTrimming=True)
    trimmedPoints = np.concatenate((trimmedPoints1, trimmedPoints2), axis=0)

    return prunedBranches, trimmedPoints


def connectComponent(connectionComponent: UnconnectedComponent,
                     unconnectedComponents: Set[UnconnectedComponent],
                     connectedComponents: Set[ConnectedComponent],
                     allBranches: Set[Branch],
                     assignedPointsLookUp: KDTree,
                     assignedPointsDistances: npt.NDArray[np.double],
                     originalBranchIndices: Dict[Branch, npt.NDArray[np.intp]],
                     resolution: float,
                     maxConnectionLength: float,
                     minAngleThreshold: float,
                     minDistanceThreshold: float,
                     ) -> Tuple[Branch, Optional[Branch]]:
    unconnectedComponents.remove(connectionComponent)
    connectionComponent, connectionBranch, splitEnd = connectionComponent.connect(assignedPointsLookUp,
                                                                                  assignedPointsDistances)
    connectedComponents.add(connectionComponent)

    interpolation, interpolationIndices = connectionBranch.getInterpolation(resolution)
    connectionBranch.skeletonPoints = interpolation

    _, originalIndices = np.unique(interpolationIndices, return_index=True)
    originalBranchIndices[connectionBranch] = originalIndices

    if splitEnd is not None:
        splitBranch = splitEnd.parent
        for component in connectedComponents:
            if splitBranch in component.directionData:
                del component.directionData[splitBranch]
        if splitBranch in originalBranchIndices:
            originalIndices = originalBranchIndices[splitBranch]
            splitIdx = len(splitBranch) - 1
            originalBranchIndices[splitEnd] = np.insert(originalIndices[originalIndices > splitIdx] - splitIdx, 0, 0)
            originalBranchIndices[splitBranch] = np.append(originalIndices[originalIndices < splitIdx], splitIdx)

    for component in unconnectedComponents:
        if component.getBoundingBoxDistance(connectionBranch[0]) <= maxConnectionLength:
            component.updateConnections(connectionBranch, minAngleThreshold, minDistanceThreshold)

    return connectionBranch, splitEnd


def findConnections(connectedComponent: ConnectedComponent,
                    unconnectedComponent: UnconnectedComponent,
                    potentialConnections: nx.Graph,
                    unassignedPointsLookUp: KDTree,
                    unassignedPointsDistances: npt.NDArray[np.double],
                    minDistanceThreshold: float,
                    maxDistanceThreshold: float,
                    minAngleThreshold: float,
                    maxAngleThreshold: float,
                    windowSize: float,
                    densityFactor: float):
    for endBranch in unconnectedComponent.branches:
        if connectedComponent.getBoundingBoxDistance(endBranch) > maxDistanceThreshold:
            continue
        directionsAtEnd = unconnectedComponent.getDirectionData(endBranch, minDistanceThreshold)

        for startBranch in connectedComponent.branches:
            if startBranch.branchBoundingBoxDistance(endBranch) > maxDistanceThreshold:
                continue
            directionsAtStart = connectedComponent.getDirectionData(startBranch, minDistanceThreshold)
            searchRange = connectedComponent.searchRanges[startBranch]

            for endIdx, startDirection, childDirections in directionsAtEnd:
                for startIdx, parentDirections, siblingDirections in directionsAtStart:
                    startPoint = startBranch[startIdx]
                    endPoint = endBranch[endIdx]
                    connectionLength = np.linalg.norm(endPoint - startPoint)

                    if (connectionLength > maxDistanceThreshold or
                            connectionLength <= connectedComponent.checkedDistanceThreshold):
                        continue

                    # score the direct connection between the points
                    connectionLine = np.stack((startPoint, endPoint))
                    startTlsPoints, startTlsDistances = connectedComponent.getTlsData(startBranch, startIdx)
                    score = scoreConnection(connectionLine, searchRange,
                                            startDirection, childDirections, parentDirections, siblingDirections,
                                            startTlsPoints, startTlsDistances,
                                            unassignedPointsLookUp, unassignedPointsDistances,
                                            minAngleThreshold, maxAngleThreshold, windowSize, densityFactor)
                    if score is not None:
                        connection = (score, startBranch, startIdx, endBranch, endIdx, connectionLine, connectedComponent)
                        unconnectedComponent.connections.append(connection)

                    try:
                        path = nx.shortest_path(potentialConnections, (startBranch, startIdx), (endBranch, endIdx),
                                                weight='distance')
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

                    connectionStrip = np.stack([branch.skeletonPoints[idx] for branch, idx in path])
                    score = scoreConnection(connectionStrip, searchRange,
                                            startDirection, childDirections, parentDirections, siblingDirections,
                                            startTlsPoints, startTlsDistances,
                                            unassignedPointsLookUp, unassignedPointsDistances,
                                            minAngleThreshold, maxAngleThreshold, windowSize, densityFactor)
                    if score is not None:
                        connection = (score, startBranch, startIdx, endBranch, endIdx, connectionStrip, connectedComponent)
                        unconnectedComponent.connections.append(connection)

    unconnectedComponent.connections.sort(key=lambda c: c[0], reverse=True)


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
