import numpy as np
import os
import pytest
from scipy.io import loadmat
from typing import Any, Dict
import yaml

from tests.utils.data_import import rootPath


@pytest.fixture(scope='module')
def base_point():
    data_path = "TestData/InputTlsPoints/base.mat"
    return loadmat(os.path.join(rootPath(), data_path))['basePoint'].squeeze()


@pytest.fixture(scope='module')
def base_radius():
    data_path = "TestData/InputTlsPoints/base.mat"
    return loadmat(os.path.join(rootPath(), data_path))['baseRadius'].squeeze()


@pytest.fixture(scope='module')
def initial_search_range(base_radius):
    return 0.1 * base_radius


@pytest.fixture(scope='module')
def parameters() -> Dict[str, Any]:
    if not hasattr(parameters, 'dict'):
        data_path = "TestData/parameters.yaml"
        with open(os.path.join(rootPath(), data_path)) as config:
            parameters.dict = yaml.safe_load(config)["parameters"]
    return parameters.dict.copy()


@pytest.fixture(scope='module')
def tls_points(base_point) -> np.ndarray:
    if not hasattr(tls_points, 'pos'):
        data_path = "TestData/InputTlsPoints/Positions.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            points = np.loadtxt(input_file, dtype=np.double)
        points = np.unique(points, axis=0)
        points = np.sort(points.view([('x', np.double), ('y', np.double), ('z', np.double)]).squeeze(-1),
                         order=['z','x','y']).view((np.double, 3))
        tls_points.pos = np.append(points, base_point[np.newaxis], axis=0)
    return tls_points.pos.copy()
