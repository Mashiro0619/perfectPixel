import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(params=["numpy", "opencv"])
def backend(request):
    if request.param == "opencv":
        pytest.importorskip("cv2")
        name = "perfect_pixel"
    else:
        name = "perfect_pixel_noCV2"
    return importlib.import_module("perfect_pixel." + name)


@pytest.fixture
def repository():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def blocks():
    def make(grid=32, cell=16):
        width, height = (grid, grid) if isinstance(grid, int) else grid
        small = np.random.default_rng(3).integers(0, 256, (height, width, 3), dtype=np.uint8)
        large = np.repeat(np.repeat(small, cell, axis=0), cell, axis=1)
        return small, large
    return make
