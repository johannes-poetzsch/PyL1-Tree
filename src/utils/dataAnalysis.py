import numpy as np
import numpy.typing as npt

from typing import Optional, Tuple


def rollingMedianSmooth(values: npt.NDArray,
                        window: int):
    """
    Rolling median smoothing with edge padding.
    Robust to single-sample spikes (occlusion, a noisy slice), unlike a rolling mean.

    Parameters
    ----------
    values : (N,) ndarray
    window : int
        Window size in samples. window <= 1 disables smoothing (returns a copy of the input unchanged).
    """
    if window <= 1:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def hysteresisScan(values: npt.NDArray,
                   triggerThreshold: float,
                   confirmThreshold: float,
                   confirmSlices: int,
                   triggerAbove: bool = True) -> Tuple[Optional[int], Optional[slice]]:
    """
    Scan `values` from index 0 upward for the first point that crosses `triggerThresh`,
    then require it to stay on the same side of `confirmThresh` for `confirmSlices` consecutive samples
    (the trigger sample itself included) before accepting it.

    This is a "confirm and stop" scan: as soon as a trigger is confirmed,
    it is returned immediately -- values further along in the array
    (e.g. a crown appearing after a confirmed trunk region) can never invalidate an already-confirmed result,
    since the scan never looks at them.

    Parameters
    ----------
    values : (N,) ndarray
    triggerThreshold : float
        Threshold that defines a candidate crossing.
    confirmThreshold : float
        Threshold the confirmation window must respect. Looser than triggerThresh in the direction that tolerates minor
        fluctuation right after the trigger (this is what makes it hysteresis rather than a single fixed threshold).
    confirmSlices : int
        Length of the confirmation window, in samples.
    triggerAbove : bool
        True: trigger when values[i] >= triggerThresh; the confirmation window must stay >= confirmThresh.
        False: trigger when values[i] < triggerThresh; the confirmation window must stay < confirmThresh.

    Returns
    -------
    index : int or None
        Index of the confirmed trigger point, or None if no crossing was ever confirmed.
    window : slice or None
        The confirmation window, or None if no crossing was confirmed.
    """
    n = len(values)
    for i in range(n):
        isTriggered = values[i] >= triggerThreshold if triggerAbove else values[i] < triggerThreshold
        if isTriggered:
            window = slice(i, min(i + confirmSlices, n))
            windowVals = values[window]
            isConfirmed = np.all(windowVals >= confirmThreshold) if triggerAbove else np.all(windowVals < confirmThreshold)
            if isConfirmed:
                return i, window

    return None, None
