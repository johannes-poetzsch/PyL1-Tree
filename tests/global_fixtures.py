import numpy as np
import os
import pytest
from typing import Any, Dict
import yaml


def rootPath():
    return os.path.dirname(globals()['__file__'])


@pytest.fixture
def base_point():
    return np.array([9.672069773168593, 31.202769419959722, 0.024145170000000], dtype=np.double)


@pytest.fixture
def base_radius():
    return 0.151689744433816


@pytest.fixture
def initial_search_range(base_radius):
    return 0.1 * base_radius


@pytest.fixture
def parameters() -> Dict[str, Any]:
    if not hasattr(parameters, 'dict'):
        data_path = "TestData/parameters.yaml"
        with open(os.path.join(rootPath(), data_path)) as config:
            parameters.dict = yaml.safe_load(config)["parameters"]
    return parameters.dict.copy()


@pytest.fixture
def rel_path_positions():
    if not hasattr(rel_path_positions, 'loc'):
        data_path = "TestData/SeedPoints/GeodesicGraph/PathLengths.txt"
        with open(os.path.join(rootPath(), data_path)) as f:
            path_lengths = np.loadtxt(f, dtype=int)
        data_path = "TestData/SeedPoints/GeodesicGraph/PathLocations.txt"
        with open(os.path.join(rootPath(), data_path)) as f:
            path_locations = np.loadtxt(f, dtype=int)
        rel_path_positions.loc = path_locations / path_lengths
    return rel_path_positions.loc.copy()


@pytest.fixture
def seed_points() -> np.ndarray:
    if not hasattr(seed_points, 'pos'):
        data_path = "TestData/SeedPoints/Positions.txt"
        with open(os.path.join(rootPath(), data_path)) as input_file:
            seed_points.pos = np.loadtxt(input_file, dtype=np.double)
    return seed_points.pos.copy()


@pytest.fixture()
def seed_rel_path_positions():
    if not hasattr(seed_rel_path_positions, 'loc'):
        with open(os.path.join(rootPath(), "TestData/SeedPoints/PathLengths.txt")) as f:
            path_lengths = np.loadtxt(f, dtype=int)
        with open(os.path.join(rootPath(), "TestData/SeedPoints/PathLocations.txt")) as f:
            path_locations = np.loadtxt(f, dtype=int)
        seed_rel_path_positions.loc = path_locations / path_lengths
    return seed_rel_path_positions.loc.copy()


@pytest.fixture
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
