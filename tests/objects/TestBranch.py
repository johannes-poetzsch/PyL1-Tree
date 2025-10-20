import copy

import cupy as cp
import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal

from src.objects.branch import Branch


@pytest.fixture(scope='module')
def branch() -> Branch:
    return Branch(id=0,
                  skeletonIDs=np.arange(3, dtype=np.intp),
                  skeletonPoints=np.arange(9, dtype=np.double).reshape((3, 3)),
                  tlsPoints=np.arange(27, dtype=np.double).reshape((9, 3)),
                  cenOrder=0)


def test_to_gpu(branch):
    assert cp.get_array_module(branch.tlsPoints) is np
    branch.toGPU()
    assert cp.get_array_module(branch.tlsPoints) is cp


def test_to_cpu(branch):
    branch.toGPU()
    assert cp.get_array_module(branch.tlsPoints) is cp
    branch.toCPU()
    assert cp.get_array_module(branch.tlsPoints) is np


class TestOrientToBifurcation:
    def test_flipsToOtherEnd(self, branch):
        test_branch = copy.deepcopy(branch)
        test_branch.orientToBifurcation(test_branch.skeletonIDs[-1])

        assert test_branch.isFlipped
        assert_array_equal(test_branch.skeletonIDs, branch.skeletonIDs[::-1])
        assert_array_equal(test_branch.skeletonPoints, branch.skeletonPoints[::-1])


    def test_doesNotFlipToCurrentEnd(self, branch):
        test_branch = copy.deepcopy(branch)
        test_branch.orientToBifurcation(test_branch.skeletonIDs[0])

        assert test_branch.isFlipped == branch.isFlipped
        assert_array_equal(test_branch.skeletonIDs, branch.skeletonIDs)
        assert_array_equal(test_branch.skeletonPoints, branch.skeletonPoints)


    def test_otherIdRaisesError(self, branch):
        with pytest.raises(ValueError):
            branch.orientToBifurcation(branch.skeletonIDs[1])


def test_interpolation(branch):
    b = copy.deepcopy(branch)
    b.skeletonPoints = np.array([[0,0,0],[3,0,0],[4,1,0]])
    interpolation, interpolationIndices = b.getInterpolation(0.8)
    expected_interpolation = np.array([[0,0,0],
                                       [0.75,0,0],
                                       [1.5,0,0],
                                       [2.25,0,0],
                                       [3,0,0],
                                       [3.5,0.5,0],
                                       [4,1,0]])
    expected_interpolationIndices = np.array([0,0,0,0,1,1,2])
    assert_array_equal(interpolation, expected_interpolation)
    assert_array_equal(interpolationIndices, expected_interpolationIndices)
