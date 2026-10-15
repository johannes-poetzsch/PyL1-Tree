"""Persistent multiprocessing implementation of component connection search.

Each process owns a shard of the candidate search and keeps its candidates
between rounds. The parent applies the globally best candidate, then broadcasts
that accepted connection so each process can update its local cache.
"""

from __future__ import annotations

from itertools import chain, count
from multiprocessing import get_context
from queue import Empty
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import traceback

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from src.connectComponents import (connectComponent, findConnections, findContinuation,
                                   generatePotentialConnections, updatePotentialConnections)
from src.objects.branch import Branch
from src.objects.connectedComponent import ConnectedComponent
from src.objects.connection import Connection
from src.objects.sharedTlsPointManager import SharedTlsPointManager
from src.objects.tlsPointManager import TlsPointManager
from src.objects.unconnectedComponent import UnconnectedComponent
from src.utils.iteration import breadthFirstSearch

if TYPE_CHECKING:
    from multiprocessing.queues import Queue


class ParallelConnection(Connection):
    """Connection that can be sent as a compact, branch-ID based payload."""

    def toDescriptor(self, endComponentId: int) -> Dict[str, Any]:
        return {
            "strip": self.strip,
            "startBranchId": self.startBranch.id,
            "startIdx": self.startIdx,
            "startComponentId": self.startComponent.branches[0].id,
            "endBranchId": self.endBranch.id,
            "endIdx": self.endIdx,
            "endComponentId": endComponentId,
            "startDirection": self.startDirection,
            "childDirections": self.childDirections,
            "minDistanceThreshold": self.minDistanceThreshold,
            "minAngleThreshold": self.minAngleThreshold,
            "maxAngleThreshold": self.maxAngleThreshold,
            "windowSize": self.windowSize,
            "densityFactor": self.densityFactor,
            "stepSize": self.stepSize,
            "score": self.score,
            "boundingBoxMin": self.boundingBoxMin,
            "boundingBoxMax": self.boundingBoxMax,
        }


    def apply(self, resolution: float) -> Tuple[Branch, Optional[Branch]]:
        if getattr(self.tlsManager, "_sharedWriteEnabled", True):
            return super().apply(resolution)

        parent, sibling = self.tlsManager.splitBranch(self.startBranch, self.startIdx)
        if parent is None:
            raise ValueError("Can not connect to trunk at idx 0.")
        if sibling is not None:
            self.startComponent.branches.append(sibling)
            self.startComponent.directionData.pop(parent, None)

        emptyIds = np.empty(0, dtype=np.intp)
        emptyDistances = np.empty(0, dtype=np.double)
        connectionBranch = self.tlsManager.createBranch(self.strip, parent, emptyIds, emptyDistances,
                                                        emptyIds, emptyDistances, resolution)
        return connectionBranch, sibling


    @classmethod
    def fromDescriptor(cls,
                       payload: Dict[str, Any],
                       branchesById: Dict[int, Branch],
                       connectedById: Dict[int, ConnectedComponent],
                       tlsManager: TlsPointManager) -> "ParallelConnection":
        connection = cls.__new__(cls)
        connection.isCredible = True
        connection.strip = payload["strip"]
        connection.tlsManager = tlsManager
        connection.startBranch = branchesById[payload["startBranchId"]]
        connection.startIdx = payload["startIdx"]
        connection.startComponent = connectedById[payload["startComponentId"]]
        connection.endBranch = branchesById[payload["endBranchId"]]
        connection.endIdx = payload["endIdx"]
        connection.startDirection = payload["startDirection"]
        connection.childDirections = payload["childDirections"]
        connection.minDistanceThreshold = payload["minDistanceThreshold"]
        connection.minAngleThreshold = payload["minAngleThreshold"]
        connection.maxAngleThreshold = payload["maxAngleThreshold"]
        connection.windowSize = payload["windowSize"]
        connection.densityFactor = payload["densityFactor"]
        connection.stepSize = payload["stepSize"]
        direction = (connection.strip[-1] - connection.strip[0])
        direction = direction / np.linalg.norm(direction)
        connection.connectionDirectionsAtStart = [direction]
        connection.connectionDirectionsAtEnd = [direction]
        if len(connection.strip) > 2:
            connection.connectionDirectionsAtStart.append(
                (connection.strip[1] - connection.strip[0]) /
                np.linalg.norm(connection.strip[1] - connection.strip[0]))
            connection.connectionDirectionsAtEnd.append(
                (connection.strip[-1] - connection.strip[-2]) /
                np.linalg.norm(connection.strip[-1] - connection.strip[-2]))
        connection.assignedTlsIds = np.empty(0, dtype=np.intp)
        connection.aNewDistances = np.empty(0, dtype=np.double)
        connection.unassignedTlsIds = np.empty(0, dtype=np.intp)
        connection.uaNewDistances = np.empty(0, dtype=np.double)
        connection.boundingBoxMin = payload.get("boundingBoxMin", np.min(connection.strip, axis=0))
        connection.boundingBoxMax = payload.get("boundingBoxMax", np.max(connection.strip, axis=0))
        connection.score = payload["score"]
        return connection


def _componentKey(component: ConnectedComponent | UnconnectedComponent) -> int:
    return component.branches[0].id


def _getWorkerMaps(connectedComponents: Set[ConnectedComponent],
                   unconnectedComponents: Set[UnconnectedComponent],
                   tlsManager: TlsPointManager,
                   unfixedWrapper: Optional[Branch]
                   ) -> Tuple[Dict[int, Branch], Dict[int, ConnectedComponent],
                              Dict[int, UnconnectedComponent]]:
    branchesById = {branch.id: branch for branch in tlsManager.branches}
    for component in chain(connectedComponents, unconnectedComponents):
        for branch in component.branches:
            branchesById[branch.id] = branch
    if unfixedWrapper is not None:
        branchesById[unfixedWrapper.id] = unfixedWrapper
    connectedById = {_componentKey(component): component for component in connectedComponents}
    unconnectedById = {_componentKey(component): component for component in unconnectedComponents}
    return branchesById, connectedById, unconnectedById


def _workerMain(workerId: int,
                state: Tuple[Any, ...],
                inputQueue: Queue[Any],
                outputQueue) -> None:
    try:
        _workerLoop(workerId, state, inputQueue, outputQueue)
    except BaseException:
        outputQueue.put((workerId, "error", traceback.format_exc()))
    finally:
        tlsManager = state[2]
        if isinstance(tlsManager, SharedTlsPointManager):
            tlsManager.closeSharedMemory()


def _workerLoop(workerId: int,
                state: Tuple[Any, ...],
                inputQueue: Queue[Any],
                outputQueue) -> None:
    (connectedComponents, unconnectedComponents, tlsManager, potentialConnections,
     unfixedLookUp, unfixedWrapper, minDistanceThreshold, branchIdStart) = state
    connectedComponents = set(connectedComponents)
    unconnectedComponents = set(unconnectedComponents)
    Branch.idGenerator = count(branchIdStart)

    while True:
        command, payload = inputQueue.get()
        if command == "stop":
            return
        if command == "search":
            for connectedId, unconnectedId, startId, endId, checkedThreshold, params in payload:
                branchesById, connectedById, unconnectedById = _getWorkerMaps(
                    connectedComponents, unconnectedComponents, tlsManager, unfixedWrapper)
                connectedComponent = connectedById[connectedId]
                unconnectedComponent = unconnectedById[unconnectedId]
                connectedComponent.checkedDistanceThreshold = checkedThreshold
                findConnections(connectedComponent, unconnectedComponent, potentialConnections, tlsManager,
                                minDistanceThreshold, params["maxDistanceThreshold"],
                                params["minAngleThreshold"], params["maxAngleThreshold"],
                                params["windowSize"], params["densityFactor"], params["stepSize"],
                                endBranches=[branchesById[endId]], startBranches=[branchesById[startId]],
                                connectionType=ParallelConnection)

            branchesById, connectedById, unconnectedById = _getWorkerMaps(
                connectedComponents, unconnectedComponents, tlsManager, unfixedWrapper)
            best = None
            for unconnectedId, component in unconnectedById.items():
                for connection in component.connections:
                    if best is None or connection.score > best[0]:
                        best = (connection.score, connection.toDescriptor(unconnectedId))
            outputQueue.put((workerId, "best", best))
            continue

        if command == "apply":
            branchesById, connectedById, unconnectedById = _getWorkerMaps(
                connectedComponents, unconnectedComponents, tlsManager, unfixedWrapper)
            connectionComponent = unconnectedById[payload["endComponentId"]]
            connection = ParallelConnection.fromDescriptor(payload, branchesById, connectedById, tlsManager)
            connectionComponent.connections.insert(0, connection)
            connectionBranch, splitEnd = connectComponent(connectionComponent, unconnectedComponents,
                                                          connectedComponents, tlsManager,
                                                          payload["resolution"])
            if isinstance(tlsManager, SharedTlsPointManager):
                tlsManager.applyBranchSlices(payload["branchSlices"], payload["changedBranchIds"])
            if unfixedWrapper is not None:
                updatePotentialConnections(potentialConnections, unfixedLookUp, unfixedWrapper,
                                           connectionBranch, splitEnd, minDistanceThreshold)
            outputQueue.put((workerId, "applied", None))


def connectComponentsParallel(trunk: Branch,
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
                              stepSize: Optional[float],
                              workers: int
                              ) -> Tuple[List[Branch], npt.NDArray[np.double]]:
    context = get_context("spawn")
    tlsManager = None
    inputQueues = []
    outputQueue = None
    processes = []
    startedProcesses = []
    try:
        rootComponent = ConnectedComponent(list(breadthFirstSearch(trunk)))
        connectedComponents = {rootComponent}
        unconnectedComponents = {UnconnectedComponent(list(breadthFirstSearch(stem)))
                                 for stem in componentStems if stem is not trunk}
        originalComponentCount = len(componentStems)
        branches = list(chain(breadthFirstSearch(trunk),
                              chain.from_iterable([component.branches for component in unconnectedComponents])))
        tlsManager = SharedTlsPointManager(branches, unassignedPoints)
        unfixedLookUp = KDTree(unfixedPoints)
        unfixedWrapper = Branch(unfixedPoints) if len(unfixedPoints) > 0 else None
        potentialConnections = generatePotentialConnections(branches, unfixedLookUp, unfixedWrapper,
                                                            minDistanceThreshold)

        # Processes receive their own persistent graph, manager, component objects,
        # and candidate cache. Only candidate work and accepted-connection deltas
        # cross the queues after startup.
        branchIdStart = next(Branch.idGenerator)
        Branch.idGenerator = count(branchIdStart)
        initialState = (list(connectedComponents), list(unconnectedComponents), tlsManager, potentialConnections,
                        unfixedLookUp, unfixedWrapper, minDistanceThreshold, branchIdStart)
        inputQueues = [context.Queue() for _ in range(workers)]
        outputQueue = context.Queue()
        processes = [context.Process(target=_workerMain,
                                     args=(workerId, initialState, inputQueues[workerId], outputQueue))
                     for workerId in range(workers)]

        originalMaxDistanceThreshold = maxDistanceThreshold
        maxConnectionLength = 2.0
        params = {"minAngleThreshold": minAngleThreshold, "maxAngleThreshold": maxAngleThreshold,
                  "windowSize": windowSize, "densityFactor": densityFactor, "stepSize": stepSize}
        for process in processes:
            process.start()
            startedProcesses.append(process)

        while unconnectedComponents:
            tasks = []
            for component in connectedComponents:
                if component.checkedDistanceThreshold >= maxDistanceThreshold:
                    continue
                for otherComponent in unconnectedComponents:
                    if component.getObjectBBDistance(otherComponent) > maxDistanceThreshold:
                        continue
                    for endBranch in otherComponent.branches:
                        if component.getObjectBBDistance(endBranch) > maxDistanceThreshold:
                            continue
                        directionsAtEnd = otherComponent.getDirectionData(endBranch, minDistanceThreshold)
                        for startBranch in component.branches:
                            if startBranch.getObjectBBDistance(endBranch) <= maxDistanceThreshold:
                                weight = max(1, (len(startBranch) - 1) * len(directionsAtEnd))
                                tasks.append((weight, _componentKey(component), _componentKey(otherComponent),
                                              startBranch.id, endBranch.id, component.checkedDistanceThreshold))
                component.checkedDistanceThreshold = maxDistanceThreshold

            # Greedy weight balancing splits even one large component across workers.
            workerTasks: List[List[Tuple[Any, ...]]] = [[] for _ in range(workers)]
            workerWeights = [0] * workers
            for task in sorted(tasks, reverse=True):
                workerId = min(range(workers), key=workerWeights.__getitem__)
                weight, connectedId, unconnectedId, startId, endId, checkedThreshold = task
                taskParams = dict(params, maxDistanceThreshold=maxDistanceThreshold)
                workerTasks[workerId].append((connectedId, unconnectedId, startId, endId,
                                              checkedThreshold, taskParams))
                workerWeights[workerId] += weight
            for workerId in range(workers):
                inputQueues[workerId].put(("search", workerTasks[workerId]))
            workerBest = _getWorkerResults(outputQueue, workers, "best", startedProcesses)
            bestPayloads = [(workerId, result) for workerId, result in enumerate(workerBest) if result is not None]
            if bestPayloads:
                _, winner = max(bestPayloads, key=lambda item: item[1][0])
                connectionPayload = winner[1]
                connectionPayload["resolution"] = resolution
                branchesById, connectedById, unconnectedById = _getWorkerMaps(
                    connectedComponents, unconnectedComponents, tlsManager, unfixedWrapper)
                connectionComponent = unconnectedById[connectionPayload["endComponentId"]]
                connection = ParallelConnection.fromDescriptor(connectionPayload, branchesById,
                                                               connectedById, tlsManager)
                connectionComponent.connections.insert(0, connection)
            else:
                if len(unconnectedComponents) <= originalComponentCount / 10:
                    break
                if maxDistanceThreshold + 0.2 <= maxConnectionLength:
                    maxDistanceThreshold += 0.2
                    continue
                connectionComponent = findContinuation(connectedComponents, unconnectedComponents,
                                                       tlsManager, maxConnectionLength)
                if connectionComponent is None:
                    break
                connection = connectionComponent.connections[0]
                connectionPayload = None

            tlsManager.resetChangedBranches()
            connectionBranch, splitEnd = connectComponent(connectionComponent, unconnectedComponents,
                                                          connectedComponents, tlsManager, resolution)
            if connectionPayload is None:
                connectionPayload = _connectionToPayload(connection, connectionComponent, resolution)
            connectionPayload["boundingBoxMin"] = connection.boundingBoxMin
            connectionPayload["boundingBoxMax"] = connection.boundingBoxMax
            if unfixedWrapper is not None:
                updatePotentialConnections(potentialConnections, unfixedLookUp, unfixedWrapper,
                                           connectionBranch, splitEnd, minDistanceThreshold)

            connectionPayload["resolution"] = resolution
            changedBranchIds = tlsManager.getChangedBranchIds()
            connectionPayload["branchSlices"] = tlsManager.exportBranchSlices(changedBranchIds)
            connectionPayload["changedBranchIds"] = changedBranchIds
            for inputQueue in inputQueues:
                inputQueue.put(("apply", connectionPayload))
            _getWorkerResults(outputQueue, workers, "applied", startedProcesses)
            maxDistanceThreshold = originalMaxDistanceThreshold

        trimmedPoints = tlsManager.releasePoints()
        for branch in reversed(list(breadthFirstSearch(trunk))):
            branch.removeTemporaryPoints()
            if len(branch.children) == 1:
                branch.expand()
        prunedBranches = list(chain.from_iterable([component.branches for component in unconnectedComponents]))
        return prunedBranches, trimmedPoints
    finally:
        for workerId, process in enumerate(startedProcesses):
            if process.is_alive():
                inputQueues[workerId].put(("stop", None))
        for process in startedProcesses:
            process.join()
        for inputQueue in inputQueues:
            inputQueue.close()
        if outputQueue is not None:
            outputQueue.close()
        if tlsManager is not None:
            tlsManager.closeSharedMemory(unlink=True)


def _connectionToPayload(connection: Connection,
                         endComponent: UnconnectedComponent,
                         resolution: float) -> Dict[str, Any]:
    payload = {"strip": connection.strip,
               "startBranchId": connection.startBranch.id,
               "startIdx": connection.startIdx,
               "startComponentId": _componentKey(connection.startComponent),
               "endBranchId": connection.endBranch.id,
               "endIdx": connection.endIdx,
               "endComponentId": _componentKey(endComponent),
               "startDirection": connection.startDirection,
               "childDirections": connection.childDirections,
               "minDistanceThreshold": connection.minDistanceThreshold,
               "minAngleThreshold": connection.minAngleThreshold,
               "maxAngleThreshold": connection.maxAngleThreshold,
               "windowSize": connection.windowSize,
               "densityFactor": connection.densityFactor,
               "stepSize": connection.stepSize,
               "score": getattr(connection, "score", 0.0),
               "boundingBoxMin": getattr(connection, "boundingBoxMin", np.min(connection.strip, axis=0)),
               "boundingBoxMax": getattr(connection, "boundingBoxMax", np.max(connection.strip, axis=0)),
               "resolution": resolution}
    return payload


def _getWorkerResults(outputQueue: Queue[Any],
                      workers: int,
                      expectedCommand: str,
                      processes: List[Any]) -> List[Any]:
    results = [None] * workers
    received = set()
    while len(received) < workers:
        try:
            workerId, command, result = outputQueue.get(timeout=0.5)
        except Empty:
            exitedWorkers = [workerId for workerId, process in enumerate(processes)
                             if workerId not in received and process.exitcode is not None]
            if exitedWorkers:
                raise RuntimeError(f"Multiprocessing workers exited without responding: {exitedWorkers}")
            continue
        if command == "error":
            raise RuntimeError(f"Multiprocessing worker {workerId} failed:\n{result}")
        if command != expectedCommand or workerId in received:
            raise RuntimeError(f"Unexpected worker response: {workerId}, {command}")
        received.add(workerId)
        results[workerId] = result
    return results
