import numpy as np
import scipy


xp = np
xsp = scipy
xdf = None
gpuEnabled = False

def enable_gpu():
    global xp, xsp, xdf, gpuEnabled

    try:
        import cupy
        import cupyx.scipy
        import cudf
    except ModuleNotFoundError as exc:
        if exc.name in {"cupy", "cupyx"}:
            raise RuntimeError(
                "GPU acceleration requires CuPy. Install the optional GPU dependencies."
            ) from exc
        raise

    xp = cupy
    xsp = cupyx.scipy
    xdf = cudf.DataFrame
    gpuEnabled = True


def disable_gpu():
    global xp, xsp, gpuEnabled

    xp = np
    xsp = scipy
    gpuEnabled = False


def to_numpy(*arrays):
    if gpuEnabled:
        converted = tuple(xp.asnumpy(array) for array in arrays)
    else:
        converted = tuple(np.asarray(array, copy=None) for array in arrays)

    return converted[0] if len(converted) == 1 else converted


def array_module(*arrays):
    if gpuEnabled:
        return xp.get_array_module(*arrays)
    return np
