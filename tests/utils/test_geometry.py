import numpy as np
import pytest
from scipy.spatial import KDTree

from src.objects.tlsPointManager import TlsPointManager
from src.utils.geometry import queryPointsNearSegment, setFastStripPointDistances, stripPointsDistances


def test_query_points_near_segment_returns_integer_indices_when_empty():
    lookup = KDTree(np.array([[10.0, 10.0, 10.0]]))

    indices = queryPointsNearSegment(
        lookup,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        0.01,
    )

    assert indices.dtype == np.dtype(np.intp)
    assert indices.size == 0


def test_find_point_indices_around_strip_handles_empty_segment_queries():
    points = np.array([[0.5, 0.0, 0.0]])
    lookup = KDTree(points)
    strip = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
    ])

    indices = TlsPointManager.findPointIndicesAroundStrip(
        strip,
        lookup,
        np.array([0.01]),
        0.1,
    )

    np.testing.assert_array_equal(indices, np.array([0], dtype=np.intp))


def test_fast_strip_distances_match_legacy_with_trimming():
    strip = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    points = np.array([
        [-0.5, 0.0, 0.0],
        [0.5, 0.2, 0.0],
        [1.0, 0.5, 0.1],
        [1.0, 1.5, 0.0],
    ])

    setFastStripPointDistances(False)
    expected_distances, expected_trimming = stripPointsDistances(
        strip,
        points,
        trimStart=True,
        trimEnd=True,
        returnTrimming=True,
        maxBatchSize=2,
    )

    setFastStripPointDistances(True)
    try:
        actual_distances, actual_trimming = stripPointsDistances(
            strip,
            points,
            trimStart=True,
            trimEnd=True,
            returnTrimming=True,
            maxBatchSize=2,
        )
        np.testing.assert_allclose(actual_distances, expected_distances, atol=1e-12)
        np.testing.assert_array_equal(actual_trimming, expected_trimming)
        np.testing.assert_allclose(
            stripPointsDistances(strip, points, trimStart=True, trimEnd=True, maxBatchSize=2),
            expected_distances[~expected_trimming],
            atol=1e-12,
        )
    finally:
        setFastStripPointDistances(False)


@pytest.mark.parametrize("use_fast", [False, True])
def test_strip_distances_reject_zero_length_segments(use_fast):
    strip = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    setFastStripPointDistances(use_fast)
    try:
        with pytest.raises(ValueError, match="must not contain zero-length segments"):
            stripPointsDistances(strip, np.array([[0.5, 0.2, 0.0]]), returnPedalInfo=True)
    finally:
        setFastStripPointDistances(False)


@pytest.mark.parametrize("use_fast", [False, True])
def test_strip_distances_reject_strip_with_only_zero_length_segments(use_fast):
    strip = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])

    setFastStripPointDistances(use_fast)
    try:
        with pytest.raises(ValueError, match="must not contain zero-length segments"):
            stripPointsDistances(strip, np.array([[0.0, 0.0, 0.0]]))
    finally:
        setFastStripPointDistances(False)
