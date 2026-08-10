import warnings

try:
    import cupy as cp
    import cupyx.scipy as csp
except ModuleNotFoundError:
    warnings.warn("Warning: cupy not available")
    import numpy as cp
    import scipy as csp
