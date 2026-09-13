import subprocess
import sys

import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize("method", ["center", "median", "majority"])
def test_exact_round_trip(backend, blocks, method):
    expected, source = blocks()
    w, h, out = backend.get_perfect_pixel(source, sample_method=method)
    assert (w, h) == (32, 32)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("name", [
    "avatar.png", "car.png", "comfyui.png", "girl.jpg", "pig.png",
    "rika.png", "robot.jpeg", "shanxi.jpg", "skull.png",
])
@pytest.mark.parametrize("method", ["center", "median", "majority"])
def test_repository_images_smoke(backend, repository, name, method):
    with Image.open(repository / "images" / name) as file:
        source = np.asarray(file.convert("RGB"))
    before = source.copy()
    w, h, out = backend.get_perfect_pixel(source, sample_method=method)
    np.testing.assert_array_equal(source, before)
    assert out.dtype == np.uint8
    if w is None:
        assert h is None and out is source
    else:
        assert w > 0 and h > 0 and out.shape == (h, w, 3)
        assert min(source.shape[1] / w, source.shape[0] / h) >= 4


@pytest.mark.parametrize("value", [0, 127, 255])
@pytest.mark.parametrize("side", [1, 16, 64, 512])
def test_uniform_images_fail_gracefully(backend, value, side):
    source = np.full((side, side, 3), value, np.uint8)
    w, h, out = backend.get_perfect_pixel(source)
    assert w is None and h is None and out is source
    assert backend.estimate_grid_fft(backend.rgb_to_gray(source)) == (None, None)


def test_fft_failure_reaches_gradient_fallback(backend, blocks, monkeypatch):
    expected, source = blocks(grid=16, cell=32)
    monkeypatch.setattr(backend, "estimate_grid_fft", lambda *a, **k: (None, None))
    w, h, out = backend.get_perfect_pixel(source)
    assert (w, h) == (16, 16)
    np.testing.assert_array_equal(out, expected)


def test_public_rgb_color_convention(backend):
    palette = np.array([[255, 0, 0], [0, 0, 97]], np.uint8)
    small = palette[np.random.default_rng(11).integers(0, 2, (32, 32))]
    source = np.repeat(np.repeat(small, 16, 0), 16, 1)
    w, h, out = backend.get_perfect_pixel(source)
    assert (w, h) == (32, 32)
    np.testing.assert_array_equal(out, small)


@pytest.mark.parametrize("grid,cell", [(16, 32), (32, 32), (8, 64)])
def test_coarse_grid_is_preserved(backend, blocks, grid, cell):
    expected, source = blocks(grid, cell)
    w, h, out = backend.get_perfect_pixel(source)
    assert (w, h) == (grid, grid)
    np.testing.assert_array_equal(out, expected)


def test_min_size_applies_to_gradient_fallback(backend, repository):
    with Image.open(repository / "images/car.png") as file:
        source = np.asarray(file.convert("RGB"))
    w, h, out = backend.get_perfect_pixel(source, min_size=64)
    if w is None:
        assert h is None and out is source
    else:
        assert min(source.shape[1] / w, source.shape[0] / h) >= 64


def test_min_size_applies_after_refinement(backend, monkeypatch):
    source = np.zeros((100, 100, 3), np.uint8)
    monkeypatch.setattr(backend, "detect_grid_scale", lambda *a, **k: (5, 5))
    monkeypatch.setattr(backend, "refine_grids", lambda *a, **k: (list(range(0, 101, 10)),) * 2)
    w, h, out = backend.get_perfect_pixel(source, min_size=20)
    assert w is None and h is None and out is source


@pytest.mark.parametrize("shape,grid", [
    ((100, 100), (1, 1)), ((100, 100), (5, 5)), ((521, 521), (3, 3)),
    ((17, 17), (3, 3)), ((7, 11), (11, 7)), ((1, 2), (2, 1)),
])
@pytest.mark.parametrize("method", ["center", "median", "majority"])
def test_manual_grids_have_exact_dimensions(backend, shape, grid, method):
    source = np.full((*shape, 3), 127, np.uint8)
    w, h, out = backend.get_perfect_pixel(
        source, grid_size=grid, min_size=1000, fix_square=True, sample_method=method)
    assert (w, h) == grid
    assert out.shape == (grid[1], grid[0], 3) and np.all(out == 127)


def test_zero_refinement_keeps_uniform_manual_spacing(backend, blocks):
    expected, source = blocks(grid=(5, 3), cell=20)
    w, h, out = backend.get_perfect_pixel(source, grid_size=(5, 3), refine_intensity=0)
    assert (w, h) == (5, 3)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("intensity", [0, 0.25, 0.5])
@pytest.mark.parametrize("grid", [(1, 1), (3, 5), (23, 17)])
@pytest.mark.parametrize("manual", [False, True])
def test_grid_lines_are_bounded_and_strictly_increasing(backend, intensity, grid, manual):
    source = np.random.default_rng(12).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    xs, ys = backend.refine_grids(source, *grid, refine_intensity=intensity, preserve_count=manual)
    for coords, length, count in ((xs, 23, grid[0]), (ys, 17, grid[1])):
        assert coords[0] == 0 and coords[-1] == length
        assert np.all(np.diff(coords) > 0)
        if manual:
            assert len(coords) == count + 1


def test_center_sampler_clips_outer_coordinates(backend):
    source = np.arange(17 * 17 * 3, dtype=np.int32).reshape(17, 17, 3)
    out = backend.sample_center(source, [-20, 10, 50], [-20, 10, 50])
    expected = source[np.array([0, 16])[:, None], np.array([0, 16])[None, :]]
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("grid", [(128, 128), (-4, 4)])
def test_dangerous_grids_are_rejected_without_hanging(backend, grid):
    code = f"""
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('subject', {backend.__file__!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.get_perfect_pixel(np.zeros((64,64,3), np.uint8), grid_size={grid!r})
except ValueError:
    print('REJECTED')
else:
    raise AssertionError('Expected ValueError')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True,
        timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stderr
    assert "REJECTED" in result.stdout


@pytest.mark.parametrize("grid", [(0, 4), (4, 0), (4.5, 4), (True, 4), (np.nan, 4), (), (4,), (4, 4, 4)])
def test_other_invalid_manual_grids(backend, grid):
    with pytest.raises(ValueError, match="grid_size"):
        backend.get_perfect_pixel(np.zeros((64, 64, 3), np.uint8), grid_size=grid)


@pytest.mark.parametrize("kwargs", [
    {"sample_method": "unknown"}, {"min_size": 0}, {"min_size": -1},
    {"min_size": np.nan}, {"min_size": np.inf}, {"min_size": True},
    {"peak_width": 0}, {"peak_width": 1.5}, {"peak_width": True},
    {"refine_intensity": -0.1}, {"refine_intensity": 0.51},
    {"refine_intensity": np.inf}, {"refine_intensity": np.nan},
])
def test_invalid_parameters_have_explicit_errors(backend, kwargs):
    with pytest.raises(ValueError):
        backend.get_perfect_pixel(np.zeros((32, 32, 3), np.uint8), **kwargs)


@pytest.mark.parametrize("source", [
    [], np.zeros((0, 32, 3)), np.zeros((32, 0, 3)), np.zeros((32, 32)),
    np.zeros((32, 32, 4)), np.ones((2, 2, 3), dtype=complex),
    np.full((2, 2, 3), np.nan), np.full((2, 2, 3), np.inf),
    np.full((2, 2, 3), 1e300), np.full((2, 2, 3), "red"),
])
def test_invalid_images_have_explicit_errors(backend, source):
    with pytest.raises(ValueError, match="image"):
        backend.get_perfect_pixel(source)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_numeric_rgb_types(backend, blocks, dtype):
    expected, source = blocks(grid=(5, 3), cell=4)
    source = source.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        source /= 255
        expected = expected.astype(dtype) / 255
    w, h, out = backend.get_perfect_pixel(source, grid_size=(5, 3), refine_intensity=0)
    assert (w, h) == (5, 3)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("values,expected", [
    ([0, 4, 4, 0], [2]), ([0, 4, 4, 4, 0], [2]),
    ([0, 4, 4.000001, 0], [2]), ([0, 0, 0], []),
])
def test_sobel_plateaus_are_single_peaks(backend, values, expected):
    np.testing.assert_array_equal(backend._gradient_peaks(np.array(values, dtype=np.float32)), expected)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 8, 16, 17, 32])
def test_smoothing_preserves_projection_length(backend, length):
    assert backend.smooth_1d(np.ones(length)).shape == (length,)


def test_default_processing_is_quiet(backend, capsys):
    backend.get_perfect_pixel(np.zeros((32, 32, 3), np.uint8))
    assert capsys.readouterr().out == ""


def test_debug_plot_is_optional_and_works_when_installed(backend, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    backend.get_perfect_pixel(np.zeros((32, 32, 3), np.uint8), grid_size=(4, 4), debug=True)
    assert shown == [True]
    plt.close("all")


@pytest.mark.parametrize("shape", [(64, 64), (512, 512), (511, 513)])
@pytest.mark.parametrize("location", ["center", "off_center"])
@pytest.mark.parametrize("values", [(255, 0), (127, 128), (1.0, 0.0)])
def test_isolated_pixel_does_not_create_a_grid(backend, shape, location, values):
    background, foreground = values
    dtype = np.float32 if isinstance(background, float) else np.uint8
    source = np.full((*shape, 3), background, dtype=dtype)
    position = (shape[0] // 2, shape[1] // 2) if location == "center" else (1, 7)
    source[position] = foreground
    assert backend.estimate_grid_fft(backend.rgb_to_gray(source)) == (None, None)
    w, h, out = backend.get_perfect_pixel(source)
    assert w is None and h is None and out is source
    assert np.all(out[position] == foreground)


@pytest.mark.parametrize("ripple", [0.0, 1e-8])
def test_dc_notch_shoulders_are_not_fft_peaks(backend, ripple):
    projection = np.ones(512)
    projection[256] = 0
    projection += ripple * np.cos(np.arange(512) * 0.17)
    assert backend.detect_peak(backend.smooth_1d(projection)) is None


def test_short_flat_topped_fft_peaks_remain_valid(backend):
    x = np.arange(128)
    projection = 0.25 + np.exp(-((x - 31.5) / 4) ** 2) + np.exp(-((x - 95.5) / 4) ** 2)
    assert backend.detect_peak(projection) == pytest.approx(32, abs=0.5)


@pytest.mark.parametrize("grid,cell", [(16, 32), (32, 16)])
def test_sparse_but_real_pixel_grid_is_preserved(backend, grid, cell):
    expected = np.full((grid, grid, 3), 255, np.uint8)
    expected[grid // 2, grid // 2] = 0
    source = expected.repeat(cell, axis=0).repeat(cell, axis=1)
    w, h, out = backend.get_perfect_pixel(source)
    assert (w, h) == (grid, grid)
    np.testing.assert_array_equal(out, expected)
