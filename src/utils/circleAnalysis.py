import numpy as np
import numpy.typing as npt

from circle_detection import Ransac
from typing import Optional, Tuple


def circumferentialCompletenessIndex(circleCenter: npt.NDArray[np.double],  # shape (2,) -> x, y
                                     circleRadius: float,
                                     points: npt.NDArray[np.double],  # shape (M, 2) -> columns: x, y
                                     numRegions: int = int(365 / 5),
                                     maxDist: Optional[float] = None) -> float:
    if len(points) == 0:
        return 0.0

    angularStepSize = 2 * np.pi / numRegions

    centeredPoints = points - circleCenter  # (M, 2)
    radii = np.linalg.norm(centeredPoints, axis=1)  # (M,)

    if maxDist is None:
        mask = (radii >= 0.7 * circleRadius) & (radii <= 1.3 * circleRadius)
    else:
        mask = np.abs(radii - circleRadius) <= maxDist

    angles = np.arctan2(centeredPoints[:, 1], centeredPoints[:, 0])  # (M,)

    # shift from [-pi, pi] to [0, 2pi)
    angles = angles + np.pi
    # guard against putting exactly 2 pi into the last bin
    angles = np.where(angles >= 2.0 * np.pi, angles - 2.0 * np.pi, angles)

    sections = np.floor(angles / angularStepSize).astype(np.int64)
    sections = sections % numRegions  # (M,)

    filledSections = np.unique(sections[mask])

    circumferentialCompletenessIndex = len(filledSections) / numRegions

    return circumferentialCompletenessIndex


def fitCircle(points: npt.NDArray[np.double],
              minPoints: int = 0,
              trimQuantile: Optional[float] = None,
              useRansac: bool = False) -> Tuple[Optional[float], Optional[npt.NDArray[np.double]]]:
    if len(points) < minPoints:
        return None, None

    if useRansac:
        estimator = Ransac(bandwidth=0.05)
        estimator.detect(points[:,:2])
        estimator.filter(max_circles=1)

        if len(estimator.circles) == 0:
            return None, None
        x, y, radius = estimator.circles[0]
        center = np.array([x, y], dtype=np.double)
    else:
        mean = points.mean(axis=0)
        centeredPoints = points - mean
        x, y = centeredPoints[:,0], centeredPoints[:,1]
        A = np.column_stack([2 * x, 2 * y, np.ones(len(points))])
        sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
        a, b, c = sol
        radius = np.sqrt(c + a ** 2 + b ** 2)
        center = np.array([mean[0] + a, mean[1] + b])

        if trimQuantile is not None:
            distances = np.linalg.norm(points[:,:2] - center, axis=1)
            residuals = np.abs(distances - radius)
            kept = residuals <= np.quantile(residuals, 1 - trimQuantile)
            if kept.sum() >= minPoints:
                return fitCircle(points[kept], minPoints, trimQuantile=None)

    centerPoint = np.insert(center, 2, np.min(points[:, 2]))
    return radius, centerPoint
