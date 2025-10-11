import cupy as cp
import numpy.typing as npt
from typing import Optional

revCumMin = cp.RawKernel(r'''
    extern "C" __global__
    void rev_cum_min(double* x) {
        int stride = 1;
        int idx = threadIdx.x;
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        for (; stride < blockDim.x; stride *= 2){
            if (idx % (2 * stride) == 0 && idx + stride < blockDim.x) {
                x[i] = min(x[i], x[i + stride]);
            }
            __syncthreads();
        }
        stride /= 2;
        for (; stride >= 1; stride /= 2) {
            if (idx % (2 * stride) == 0 && idx - stride > 0) {
                x[i - stride] = min(x[i - stride], x[i]);
            }
            __syncthreads();
        }
    }
    ''', 'rev_cum_min')

def cum_min(x: npt.NDArray[cp.double], 
            axis: int = 0,
            reversed: bool = False):
    if axis == 0:
        x = x.transpose()
    elif not axis == 1:
        raise ValueError("CumMin axis must be 0 or 1")
    
    if reversed:
        x = x.copy()
    else:
        x = cp.fliplr(x).copy()

    if x.shape[1] > 1024:
        raise ValueError("Current CumMin implementation can only handle up to 1024 elements along the target axis")
    revCumMin((x.shape[0],), (x.shape[1],), x)

    if not reversed:
        x = cp.fliplr(x)

    if axis == 0:
        x = x.transpose()
    return x
