from collections import deque
from typing import Generator

from ..objects.branch import Branch

def breadthFirstSearch(stem: Branch, returnLeaves: bool = True) -> Generator[Branch, None, None]:
    remainingBranches = deque()
    remainingBranches.append(stem)

    while remainingBranches:
        branch: Branch = remainingBranches.pop()
        remainingBranches.extend(branch.children)
        if returnLeaves or len(branch.children) > 0:
            yield branch
