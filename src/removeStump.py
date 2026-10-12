import numpy as np
import numpy.typing as npt
from typing import Any, Dict, Optional, Tuple

from src.utils.circleAnalysis import circumferentialCompletenessIndex, fitCircle
from src.utils.dataAnalysis import hysteresisScan, rollingMedianSmooth
from src.utils.geometry import pointSort


def findBaseWithoutStump(tlsPoints: npt.NDArray[np.double],
                         resolution: float,
                         useRansac: bool = False,
                         referenceQuantile: int = 0.8,
                         highThresholdFraction: float = 0.85,
                         lowThresholdFraction: float = 0.8,
                         smoothWindow: int = 3,
                         comfirmSlices: int = 8,
                         analytics: Optional[Dict[str, Any]] = None
                         ) -> Tuple[Optional[npt.NDArray[np.double]], Optional[float], npt.NDArray[np.double]]:
    minHeight = np.min(tlsPoints[:,2])
    treeHeight = np.quantile(tlsPoints[:, 2], 0.99) - minHeight
    bottomPoints = tlsPoints[tlsPoints[:,2] < minHeight + 0.3 * treeHeight]

    heights, cci, radii, centerPoints = computeSliceProfile(bottomPoints, 2 * resolution, resolution,
                                                            trimQuantile=0.05, useRansac=useRansac)
    smoothedCci = rollingMedianSmooth(cci, smoothWindow)
    smoothedRadii = rollingMedianSmooth(radii, smoothWindow)

    reference = np.quantile(smoothedCci, referenceQuantile)
    highThreshold = highThresholdFraction * reference
    lowThreshold = lowThresholdFraction * highThreshold

    minIdx, window = hysteresisScan(smoothedCci, highThreshold, lowThreshold, comfirmSlices, triggerAbove=True)
    newMinHeight = minHeight if minIdx is None else heights[minIdx]
    if newMinHeight is None:
        return None, None, tlsPoints

    removedHeight = newMinHeight - minHeight
    remainingTlsPoints = tlsPoints[tlsPoints[:,2] > newMinHeight]
    if analytics is not None:
        analytics["stump_removed_height_m"] = removedHeight
        analytics["stump_removed_point_count"] = len(tlsPoints) - len(remainingTlsPoints)

    smoothedRadii, centerPoints = smoothedRadii[minIdx:], centerPoints[minIdx:]
    baseRadius = smoothedRadii[np.argmin(np.isnan(smoothedRadii))]
    basePoint = centerPoints[np.argmin(np.any(np.isnan(centerPoints), axis=1))]

    print("Removed bottom", removedHeight, "m from the base.")
    return basePoint, baseRadius, remainingTlsPoints


def computeSliceProfile(points: npt.NDArray[np.double],
                        sliceThickness: float = 0.06,
                        sliceStep: float = 0.03,
                        numRegions: int= int(265/5),
                        minPointsPerSlice: int = 10,
                        trimQuantile: Optional[float] = None,
                        useRansac: bool = False
                        ) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    points = pointSort(points, order=('z', 'x', 'y'))
    minHeight, maxHeight = points[0,2], points[-1,2]
    heightDiff = maxHeight - minHeight
    heights, cci, radii, centerPoints = [], [], [], []

    sliceCount = int((heightDiff - sliceThickness) / sliceStep)
    for i in range(sliceCount):
        sliceHeight = minHeight + i * sliceStep
        heights.append(sliceHeight)
        minIndex = np.searchsorted(points[:, 2], sliceHeight)
        maxIdx = np.searchsorted(points[:, 2], sliceHeight + sliceThickness)
        slicePoints = points[minIndex:maxIdx]
        radius, centerPoint = fitCircle(slicePoints, minPointsPerSlice, trimQuantile, useRansac=useRansac)
        if radius is None:
            cci.append(0.0)
            radii.append(np.nan)
            centerPoints.append(np.full(3, np.nan))
        else:
            cci.append(circumferentialCompletenessIndex(centerPoint[:2], radius, slicePoints[:,:2], numRegions))
            radii.append(radius)
            centerPoints.append(centerPoint)

    return np.array(heights), np.array(cci), np.array(radii), np.stack(centerPoints)
