import importlib
import importlib.util
import shutil
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def load_node(path, name):
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return sys.modules[name + ".nodes_perfect_pixel"]


@pytest.fixture
def node(repository):
    return load_node(repository / "integrations/comfyui/PerfectPixelComfy", "test_comfy_package")


def as_tensor(source):
    return torch.from_numpy(source.copy()).float().unsqueeze(0) / 255


@pytest.mark.parametrize("selection", ["Auto", "OpenCV Backend", "Lightweight Backend"])
def test_installed_backends(node, selection):
    if selection == "OpenCV Backend":
        pytest.importorskip("cv2")
    function = node._load_backend(selection)
    if selection == "Lightweight Backend" or importlib.util.find_spec("cv2") is None:
        assert function.__module__ == "perfect_pixel.perfect_pixel_noCV2"
    else:
        assert function.__module__ == "perfect_pixel.perfect_pixel"


def test_auto_without_cv2_uses_numpy(node, monkeypatch):
    original = importlib.import_module

    def importing(name, package=None):
        if name == "cv2":
            raise ModuleNotFoundError("cv2 is not installed", name="cv2")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", importing)
    assert node._load_backend("Auto").__module__ == "perfect_pixel.perfect_pixel_noCV2"
    with pytest.raises(ModuleNotFoundError, match="cv2"):
        node._load_backend("OpenCV Backend")


@pytest.mark.parametrize("error", [
    RuntimeError("broken cv2"), ImportError("broken DLL"),
    ModuleNotFoundError("missing cv2 dependency", name="cv2_dependency"),
])
def test_auto_does_not_hide_broken_installations(node, monkeypatch, error):
    original = importlib.import_module

    def importing(name, package=None):
        if name == "cv2":
            raise error
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", importing)
    with pytest.raises(type(error), match=str(error)):
        node._load_backend("Auto")


def test_missing_internal_dependency_is_not_a_missing_backend(node, monkeypatch):
    original = importlib.import_module

    def importing(name, package=None):
        if name == ".perfect_pixel_noCV2":
            raise ModuleNotFoundError("broken copied backend", name="internal_dependency")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", importing)
    with pytest.raises(ModuleNotFoundError, match="broken copied backend"):
        node._load_backend("Lightweight Backend")


@pytest.mark.parametrize("selection", ["Auto", "OpenCV Backend", "Lightweight Backend"])
def test_copied_backends_work_without_installed_package(repository, tmp_path, monkeypatch, selection):
    if selection == "OpenCV Backend":
        pytest.importorskip("cv2")
    destination = tmp_path / "PerfectPixelComfy"
    destination.mkdir()
    for filename in ("__init__.py", "nodes_perfect_pixel.py"):
        shutil.copyfile(repository / "integrations/comfyui/PerfectPixelComfy" / filename, destination / filename)
    for filename in ("perfect_pixel.py", "perfect_pixel_noCV2.py"):
        shutil.copyfile(repository / "src/perfect_pixel" / filename, destination / filename)
    # Match ComfyUI's path-based package name rather than relying on sys.path.
    name = str(destination).replace(".", "_x_")
    copied_node = load_node(destination, name)
    original = importlib.import_module

    def importing(module, package=None):
        if module == "perfect_pixel" or module.startswith("perfect_pixel."):
            raise AssertionError("A copied backend must not need the pip package")
        return original(module, package)

    monkeypatch.setattr(importlib, "import_module", importing)
    function = copied_node._load_backend(selection)
    assert function.__module__.startswith(name + ".")
    source = np.full((32, 32, 3), 127, np.uint8)
    assert function(source, grid_size=(4, 4))[:2] == (4, 4)


@pytest.mark.parametrize("sampling", ["Center Sample", "Majority Cluster"])
@pytest.mark.parametrize("scale", [1, 4, 16])
@pytest.mark.parametrize("selection", ["Auto", "Lightweight Backend"])
def test_node_image_conversion_sampling_and_scaling(node, blocks, sampling, scale, selection):
    expected, source = blocks()
    out, = node.PerfectPixelNode().run(as_tensor(source), sampling, scale, selection)
    assert tuple(out.shape) == (1, 32 * scale, 32 * scale, 3)
    assert out.dtype == torch.float32 and 0 <= float(out.min()) <= float(out.max()) <= 1
    enlarged = np.repeat(np.repeat(expected, scale, 0), scale, 1)
    np.testing.assert_allclose(out.numpy()[0], enlarged.astype(np.float32) / 255)


def test_homogeneous_batch(node, blocks):
    _, source = blocks()
    batch = torch.from_numpy(np.stack([source, source])).float() / 255
    out, = node.PerfectPixelNode().run(batch, "Center Sample", 1, "Auto")
    assert tuple(out.shape) == (2, 32, 32, 3)


def test_mixed_grids_in_equal_sized_inputs_have_clear_error(node, blocks):
    _, first = blocks(grid=32, cell=16)
    _, second = blocks(grid=16, cell=32)
    batch = torch.from_numpy(np.stack([first, second])).float() / 255
    with pytest.raises(ValueError, match="equal input dimensions do not guarantee equal grids"):
        node.PerfectPixelNode().run(batch, "Center Sample", 1, "Auto")


def test_detection_failure_keeps_original_image(node):
    source = torch.full((1, 32, 32, 3), 1.0)
    out, = node.PerfectPixelNode().run(source, "Center Sample", 1, "Lightweight Backend")
    assert torch.equal(out, source)


@pytest.mark.parametrize("source", [
    torch.empty((0, 32, 32, 3)), torch.empty((1, 0, 32, 3)),
    torch.zeros((1, 32, 32, 4)), torch.zeros((32, 32, 3)),
    torch.full((1, 2, 2, 3), float("nan")),
])
def test_invalid_image_tensor(node, source):
    with pytest.raises(ValueError, match="IMAGE"):
        node.PerfectPixelNode().run(source, "Center Sample", 1, "Lightweight Backend")


@pytest.mark.parametrize("sampling,scale,selection", [
    ("unknown", 1, "Auto"), ("Center Sample", 0, "Auto"),
    ("Center Sample", 17, "Auto"), ("Center Sample", 1.5, "Auto"),
    ("Center Sample", True, "Auto"), ("Center Sample", 1, "unknown"),
])
def test_invalid_node_parameters(node, sampling, scale, selection):
    with pytest.raises(ValueError):
        node.PerfectPixelNode().run(torch.zeros((1, 2, 2, 3)), sampling, scale, selection)
