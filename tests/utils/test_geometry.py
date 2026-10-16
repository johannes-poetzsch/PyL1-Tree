import numpy as np
import pytest
from scipy.spatial import KDTree

from src.objects.tlsPointManager import TlsPointManager
from src.utils.geometry import (linesPointsDistances, setFastStripPointDistances, stripPointsDistances)


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


@pytest.mark.parametrize("broadcast_end_point", [False, True])
def test_fast_lines_points_distances_matches_reference(broadcast_end_point):
    line_starts = np.array([
        [100000.0, -200000.0, 300000.0],
        [100001.0, -200000.0, 300000.0],
        [100000.0, -199999.0, 300000.0],
    ])
    line_ends = (np.array([100000.0, -199999.0, 300000.0]) if broadcast_end_point else np.array([
        [100001.0, -200000.0, 300000.0],
        [100001.0, -199999.0, 300000.0],
        [100000.0, -199999.0, 300001.0],
    ]))
    points = np.array([
        [100000.3, -199999.8, 300000.2],
        [100000.5, -199999.7, 300000.3],
        [100000.8, -199999.6, 300000.1],
    ])

    setFastStripPointDistances(False)
    expected = linesPointsDistances(line_starts, line_ends, points)

    setFastStripPointDistances(True)
    try:
        actual = linesPointsDistances(line_starts, line_ends, points)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
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
